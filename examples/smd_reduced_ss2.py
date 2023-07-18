import cvxpy as cp
import numpy as np
import time
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative
import copy
import math
import pickle

# Constants for indexing state variables
X_IDX = 0
V_IDX = 1
T_IDX = 2
SINX_IDX = 3
COSX_IDX = 4


def is_psd(x):
    return np.all(np.linalg.eigvals(x) >= 0)


class escape_time_solver:

    def __init__(self, d, M, max_degree, scs_scale, scs_eps, sys_params, pcs_m, pcs_b, pcs_equality, y0, q_alpha=5, max_opt_iters=5000,
                 acc_lookback=5, solver_verbose=True, exit_time_order=1, solver='SCS'):
        self.d = d  # Dimension of system
        self.M = M  # Number of moments (multi-indexes)
        self.exit_time_order = exit_time_order
        self.solver = solver

        # Generate list of moment indexes
        idx_list = generate_unsorted_idx_list(M=max_degree, d=d)
        self.idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
        self.idx_list = self.idx_list[:M]

        # Moment arrays
        self.mj = [cp.Variable() for i in range(len(self.idx_list))]
        self.bj = [cp.Variable() for i in range(len(self.idx_list))]

        # Moment matrices
        self.moment_matrix_mj = None
        self.moment_matrix_bj = None

        # Localizing matrices
        self.localising_matrices_mj = None
        self.localising_matrices_bj = None

        # Moment idx dictionary (Used for creating LMs)
        self.mm_dict = {}  # Key: Str(i_j)
        # Eg:  entry 1,3 -> '1_3'
        # Value: moment index (multi-index)

        # Highest degree of monomials for polynomial q
        self.q_alpha = q_alpha

        # Polynomials used to define localizing matrices
        self.polynomial_coeffs_m = pcs_m
        self.polynomial_coeffs_b = pcs_b
        self.polynomial_coeffs_equality = pcs_equality

        self.y0 = y0

        # Solver parameters
        self.solver_verbose = solver_verbose
        self.max_opt_iters = max_opt_iters
        self.acc_lookback = acc_lookback
        self.scs_scale = scs_scale
        self.scs_eps = scs_eps

        # System parameters
        self.ks = sys_params[0]
        self.ms = sys_params[1]
        self.kc = sys_params[2]
        self.g = 9.81

        # Convex opt problem
        self.prob = None

    # Compile and solve problem
    # -----------------------------------------------------------------
    # Inputs:
    # 	mode - Maximize or Minimize
    # Returns:
    #  	None

    def solve(self, mode='Maximize'):
        constraints = []

        start_time = time.time()
        self.create_moment_matrix()
        print('Time to create moment matrix:', time.time()-start_time)

        start_time = time.time()
        constraints += self.mm_constraints()
        print('Time to create mm constraints:', time.time()-start_time, '\n')

        start_time = time.time()
        self.create_local_matrix()
        print('Number of localising matrices (MJ):',
              len(self.localising_matrices_mj))
        print('Number of localising matrices (BJ):',
              len(self.localising_matrices_bj))
        print('Time to create localising matrix:',
              time.time()-start_time, '\n')

        start_time = time.time()
        constraints += self.lm_constraints()
        print('Time to create lm constraints:', time.time()-start_time, '\n')

        start_time = time.time()
        constraints += self.martingale_constraints()
        print('Time to create martingale constraints:',
              time.time()-start_time, '\n')

        start_time = time.time()
        constraints += self.exit_measure_equality_constraints()
        print('Time to create localising matrix equality constraints:',
              time.time()-start_time, '\n')

        print('Number of total constraints: ', len(constraints), '\n')

        start_time = time.time()
        obj = cp.Maximize(self.exit_time_order*self.mj[self.idx_list.index([0, 0, self.exit_time_order-1, 0, 0])]) if mode == 'Maximize' else cp.Minimize(
            self.exit_time_order*self.mj[self.idx_list.index([0, 0, self.exit_time_order-1, 0, 0])])
        print('Objective', obj)
        print('Time to create objective:', time.time()-start_time)

        start_time = time.time()
        self.prob = cp.Problem(obj, constraints)
        print('Time to create problem object:', time.time()-start_time)

        if self.solver == 'SCS':
            self.prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.SCS,
                            acceleration_lookback=self.acc_lookback, eps=self.scs_eps, scale=scs_scale)
        elif self.solver == 'GUROBI':
            self.prob.solve(verbose=self.solver_verbose, solver=cp.GUROBI)
        else:
            self.prob.solve(verbose=self.solver_verbose, solver=cp.MOSEK)

        print('MJs')
        for mj in self.mj[:5]:
            print(mj.value)
        print('\nBJs')
        for bj in self.bj[:5]:
            print(bj.value)

        print(' ')

        print(self.prob.value)
        print(self.prob.status)

        return self.prob.value

    # Find the number of moment matrices possible given the provided moments
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	Maxmimum number of moment matrices

    def num_mms(self):
        for mm_idx in range(self.M):
            idx_list = generate_unsorted_idx_list(M=mm_idx, d=self.d)
            idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
            if len(idx_list) > len(self.idx_list):
                return mm_idx
            for i in range(len(idx_list)):
                for j in range(len(idx_list)):
                    moment_idx = [sum(x) for x in zip(
                        self.idx_list[i], self.idx_list[j])]
                    if moment_idx not in self.idx_list:
                        return mm_idx
        return mm_idx+1

    # Create variables for the moment matrix
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	None

    def create_moment_matrix(self):
        mm_degree = self.num_mms()-1
        idx_list = generate_unsorted_idx_list(M=mm_degree, d=self.d)
        self.moment_matrix_mj = cp.Variable(
            (len(idx_list), len(idx_list)), PSD=True)
        self.moment_matrix_bj = cp.Variable(
            (len(idx_list), len(idx_list)), PSD=True)

    # Check if a given size of localising matrix is valid
    # -----------------------------------------------------------------
    # Inputs:
    # 	q_poly_multi_idx - multi index of polynomial to be added
    #   lm_dim - dimension of localising matrix to be checked
    # Returns:
    #  	True if localising matrix can be made with defined moments

    def is_lm_valid(self, q_poly_multi_idx, lm_dim):
        for i in range(lm_dim):
            for j in range(lm_dim):
                mm_moment = [sum(x) for x in zip(
                    self.idx_list[i], self.idx_list[j])]
                lm_entry = [sum(x) for x in zip(mm_moment, q_poly_multi_idx)]
                if not lm_entry in self.idx_list:
                    return False
        return True

    # Create variables for localising matrices
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	None

    def create_local_matrix(self):

        self.localising_matrices_mj = []
        self.localising_matrices_bj = []

        # Create empty localizing matrices for MJs
        num_constraint_polys_mj = len(self.polynomial_coeffs_m)
        for j in range(num_constraint_polys_mj):
            # Generate list of moment indexes for q alpha
            idx_list = generate_unsorted_idx_list(M=self.q_alpha, d=self.d)
            idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
            # Find the largest index in the q polynomial coeff that isn't 0
            for k in range(len(idx_list)-1, -1, -1):
                if self.polynomial_coeffs_m[j][k] != 0:
                    break

            # Find the largest localising matrix that can be support by the defined moment sequence
            for i in range(self.M):
                if not self.is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=self.d))):
                    break

            # Allocate space for local matrix
            lm_dim = len(generate_unsorted_idx_list(M=i-1, d=self.d))
            self.localising_matrices_mj.append(
                cp.Variable((lm_dim, lm_dim), PSD=True))

        # Create empty localizing matrices for BJs
        num_constraint_polys_bj = len(self.polynomial_coeffs_b)
        for j in range(num_constraint_polys_bj):
            # Generate list of moment indexes for q alpha
            idx_list = generate_unsorted_idx_list(M=self.q_alpha, d=self.d)
            idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
            # Find the largest index in the q polynomial coeff that isn't 0
            for k in range(len(idx_list)-1, -1, -1):
                if self.polynomial_coeffs_b[j][k] != 0:
                    break

            # Find the largest localising matrix that can be support by the defined moment sequence
            for i in range(self.M):
                if not self.is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=self.d))):
                    break

            # Allocate space for local matrix
            lm_dim = len(generate_unsorted_idx_list(M=i-1, d=self.d))
            self.localising_matrices_bj.append(
                cp.Variable((lm_dim, lm_dim), PSD=True))

    # Assign moment matrix entries to moments
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	List of constraints that assign moments to moment matrices

    def mm_constraints(self):
        constraints = []

        vis_mm = {}

        for i in range(self.moment_matrix_mj.shape[0]):
            for j in range(self.moment_matrix_mj.shape[1]):
                moment_idx = [sum(x) for x in zip(
                    self.idx_list[i], self.idx_list[j])]

                constraints.append(self.moment_matrix_mj[i][j]
                                   == self.mj[self.idx_list.index(moment_idx)])
                constraints.append(self.moment_matrix_bj[i][j]
                                   == self.bj[self.idx_list.index(moment_idx)])

                # Create dictionary entry for LMs
                self.mm_dict[str(i)+'_'+str(j)] = moment_idx

                # Add to dictionary for visualisation
                vis_mm[(i, j)] = moment_idx

        # Visualise moment matrix
        mm = [[None for _ in range(self.moment_matrix_mj.shape[1])]
              for _ in range(self.moment_matrix_mj.shape[0])]
        for key in vis_mm:
            i = key[0]
            j = key[1]
            entry = vis_mm[key]
            mm[i][j] = entry
        # print('\nMoment matrix:')
        # for row in mm:
        #     print(row)
        # print(' ')

        return constraints

    # Assign moment matrix entries to moments
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	List of constraints that assign moments to localising matrices

    def lm_constraints(self):
        constraints = []

        # Obtain the number of localising matrices for MJs
        num_lm_mj = len(self.localising_matrices_mj)
        for cur_lm in range(num_lm_mj):
            # Link to moments constraints
            for i in range(self.localising_matrices_mj[cur_lm].shape[0]):
                for j in range(self.localising_matrices_mj[cur_lm].shape[1]):
                    # Get moment index from moment matrix
                    mm_idx = self.mm_dict[str(i)+'_'+str(j)]

                    # list of alpha indexes
                    idx_list = generate_unsorted_idx_list(
                        M=self.q_alpha, d=self.d)
                    idx_list = sorted(
                        idx_list, key=cmp_to_key(index_comparison))

                    # Build constraint instruction as a string
                    constriant_str_mj = ''
                    constriant_str_mj += 'self.localising_matrices_mj['+str(
                        cur_lm)+']['+str(i)+']['+str(j)+']=='

                    for alpha in range(len(idx_list)):
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
                        if self.polynomial_coeffs_m[cur_lm][alpha] != 0:
                            # Generate constraint string
                            constriant_str_mj += 'self.polynomial_coeffs_m[cur_lm]['+str(
                                alpha)+']*'
                            constriant_str_mj += 'self.mj[' + \
                                str(self.idx_list.index(lm_idx))+']+'

                    # Remove ending '+'
                    constriant_str_mj = constriant_str_mj[:-1]
                    # Append constraint to list and execute
                    command_mj = 'constraints.append('+constriant_str_mj+')'
                    exec(command_mj)

        # Obtain the number of localising matrices for BJs
        num_lm_bj = len(self.localising_matrices_bj)
        for cur_lm in range(num_lm_bj):
            # Link to moments constraints
            for i in range(self.localising_matrices_bj[cur_lm].shape[0]):
                for j in range(self.localising_matrices_bj[cur_lm].shape[1]):
                    # Get moment index from moment matrix
                    mm_idx = self.mm_dict[str(i)+'_'+str(j)]

                    # list of alpha indexes
                    idx_list = generate_unsorted_idx_list(
                        M=self.q_alpha, d=self.d)
                    idx_list = sorted(
                        idx_list, key=cmp_to_key(index_comparison))

                    # Build constraint instruction as a string
                    constriant_str_bj = ''
                    constriant_str_bj += 'self.localising_matrices_bj['+str(
                        cur_lm)+']['+str(i)+']['+str(j)+']=='

                    for alpha in range(len(idx_list)):
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
                        if self.polynomial_coeffs_b[cur_lm][alpha] != 0:
                            # Generate constraint string
                            constriant_str_bj += 'self.polynomial_coeffs_b[cur_lm]['+str(
                                alpha)+']*'
                            constriant_str_bj += 'self.bj[' + \
                                str(self.idx_list.index(lm_idx))+']+'

                    # Remove ending '+'
                    constriant_str_bj = constriant_str_bj[:-1]
                    # Append constraint to list and execute
                    command_bj = 'constraints.append('+constriant_str_bj+')'
                    exec(command_bj)

        return constraints

    # Generate exit measure equality constraints
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	List of constraints that represent equality constraints dervied from exit measure localising matrices

    def exit_measure_equality_constraints(self):

        # return []

        constraints = []

        max_poly_degree = self.q_alpha

        # Generate list of indexes for polynomial
        idx_list = generate_unsorted_idx_list(M=max_poly_degree, d=self.d)
        idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
        # Find the largest index in the polynomial coeff that isn't 0
        for k in range(len(idx_list)-1, -1, -1):
            if self.polynomial_coeffs_equality[k] != 0:
                break

        # Find the largest localising matrix that can be support by the defined moment sequence
        for i in range(self.M):
            if not self.is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=self.d))):
                break
        lm_dim = len(generate_unsorted_idx_list(M=i-1, d=self.d))

        # Construct localising matrix entries and set to zero (equality constriants)
        # Link to moments constraints
        for i in range(lm_dim):
            for j in range(lm_dim):
                # Get moment index from moment matrix
                mm_idx = self.mm_dict[str(i)+'_'+str(j)]

                # Build constraint instruction as a string
                constriant_str = ''
                constriant_str += '0 =='

                for alpha in range(len(idx_list)):
                    # Get moment index by summing alpha index with moment matrix index
                    lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
                    if self.polynomial_coeffs_equality[alpha] != 0:
                        # Generate constraint string
                        constriant_str += 'self.polynomial_coeffs_equality['+str(
                            alpha)+']*'
                        constriant_str += 'self.bj[' + \
                            str(self.idx_list.index(lm_idx))+']+'

                # Remove ending '+'
                constriant_str = constriant_str[:-1]
                # Append constraint to list and execute
                command = 'constraints.append('+constriant_str+')'
                exec(command)

        print('Num localising matrix equality constraints:', len(constraints))
        return constraints

    # Martingale constraints
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	List of constraints that represent martingale constraints

    def martingale_constraints(self):

        constraints = []

        ################ Degree 0 #########################################################
        constraints.append(-self.bj[0] + 1 == 0)
        constraints.append(self.mj[0] >= 0)

        ################ Degree 1 #########################################################
        start = next(i for i, v in enumerate(self.idx_list)
                     if np.sum(self.idx_list[i]) == 1)
        end = next(i for i, v in enumerate(self.idx_list)
                   if np.sum(self.idx_list[i]) == 2)

        for n in range(start, end):
            invalid_idx = False
            f = copy.deepcopy(self.idx_list[n])  # Get the test function

            # For each test function perform the following derivatives:
            # dx, dv, dt, dv^2, dsin(x1), dcos(x1)
            dx = mono_derivative(f, [0])
            dv = mono_derivative(f, [1])
            dt = mono_derivative(f, [2])
            dsinx = mono_derivative(f, [3])
            dcosx = mono_derivative(f, [4])
            dv_2 = mono_derivative(f, [1, 1])

            constriant_str = ''

            # Bj constraint
            constriant_str += '-self.bj[self.idx_list.index(f)]+'

            # y0 constraint: f(y0)
            y0_const = 1.0
            for i in range(len(f)):
                if f[i] != 0:
                    y0_const *= self.y0[i]**f[i]
            constriant_str += 'y0_const+'

            # Mj constraints
            if dx[0] != 0:  # Check coefficient of derivative
                # generator term: v * df/dx
                dx[1][V_IDX] += 1  # Increment degree of monomial
                if not dx[1] in self.idx_list:
                    invalid_idx = True
                constriant_str += 'dx[0]*'
                constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'
            if dv[0] != 0:  # Check coefficient of derivative
                # generator term: (-ks*x/ms - g + kc*v*sin(x)/ms) * df/dv
                t1 = copy.copy(dv[1])
                t2 = copy.copy(dv[1])
                t3 = copy.copy(dv[1])
                t1[X_IDX] += 1
                t3[SINX_IDX] += 1
                t3[V_IDX] += 1
                if not t1 in self.idx_list or not t3 in self.idx_list:
                    invalid_idx = True
                constriant_str += '-self.ks*dv[0]/self.ms*'
                constriant_str += 'self.mj[self.idx_list.index(t1)]+'
                constriant_str += '-self.g*dv[0]*'
                constriant_str += 'self.mj[self.idx_list.index(t2)]+'
                constriant_str += 'self.kc*dv[0]/self.ms*'
                constriant_str += 'self.mj[self.idx_list.index(t3)]+'
            if dt[0] != 0:
                # generator term: df/dt
                constriant_str += '0.1*dt[0]*'
                constriant_str += 'self.mj[self.idx_list.index(dt[1])]+'
            if dsinx[0] != 0:
                # generator term: cos(x) * v * df/dsin(x)
                dsinx[1][V_IDX] += 1
                dsinx[1][COSX_IDX] += 1
                if not dsinx[1] in self.idx_list:
                    invalid_idx = True
                constriant_str += 'dsinx[0]*'
                constriant_str += 'self.mj[self.idx_list.index(dsinx[1])]+'
            if dcosx[0] != 0:
                # generator term: -sin(x) * v * df/dcos(x)
                dcosx[1][V_IDX] += 1
                dcosx[1][SINX_IDX] += 1
                if not dcosx[1] in self.idx_list:
                    invalid_idx = True
                constriant_str += '-dcosx[0]*'
                constriant_str += 'self.mj[self.idx_list.index(dcosx[1])]+'

            constriant_str = constriant_str[:-1] 	# Remove last '+' sign
            constriant_str += '==0'
            command = 'constraints.append('+constriant_str+')'
            if not invalid_idx:
                exec(command)

                ################ Degree 2 and above ###############################################
                # Get position in index list where monomial degree is 2
                start = next(i for i, v in enumerate(self.idx_list)
                             if np.sum(self.idx_list[i]) == 2)

                # n represents the index of the monomial test function
                for n in range(start, len(self.mj)):
                    invalid_idx = False
                    # Get the test function
                    f = copy.deepcopy(self.idx_list[n])

                    # For each test function perform the following derivatives:
                    # dx, dv, dv^2
                    dx = mono_derivative(f, [0])
                    dv = mono_derivative(f, [1])
                    dt = mono_derivative(f, [2])
                    dsinx = mono_derivative(f, [3])
                    dcosx = mono_derivative(f, [4])
                    dv_2 = mono_derivative(f, [1, 1])

                    constriant_str = ''

                    # Bj constraint
                    constriant_str += '-self.bj[self.idx_list.index(f)]+'

                    # y0 constraint: f(y0)
                    y0_const = 1.0
                    for i in range(len(f)):
                        if f[i] != 0:
                            y0_const *= self.y0[i]**f[i]
                    constriant_str += 'y0_const+'

                    # Mj constraints
                    if dx[0] != 0:  # Check coefficient of derivative
                        # generator term: v * df/dx
                        dx[1][V_IDX] += 1  # Increment degree of monomial
                        if not dx[1] in self.idx_list:
                            invalid_idx = True
                        constriant_str += 'dx[0]*'
                        constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'
                    if dv[0] != 0:
                        # generator term: (-ks*x/ms - g + kc*v*sin(x)/ms) * df/dv
                        t1 = copy.copy(dv[1])
                        t2 = copy.copy(dv[1])
                        t3 = copy.copy(dv[1])
                        t1[X_IDX] += 1
                        t3[SINX_IDX] += 1
                        t3[V_IDX] += 1
                        if not t1 in self.idx_list or not t3 in self.idx_list:
                            invalid_idx = True
                        constriant_str += '-self.ks*dv[0]/self.ms*'
                        constriant_str += 'self.mj[self.idx_list.index(t1)]+'
                        constriant_str += '-self.g*dv[0]*'
                        constriant_str += 'self.mj[self.idx_list.index(t2)]+'
                        constriant_str += 'self.kc*dv[0]/self.ms*'
                        constriant_str += 'self.mj[self.idx_list.index(t3)]+'
                    if dt[0] != 0:
                        # generator term: df/dt
                        constriant_str += '0.1*dt[0]*'
                        constriant_str += 'self.mj[self.idx_list.index(dt[1])]+'
                    if dsinx[0] != 0:
                        # generator term: cos(x) * v * df/dsin(x)
                        dsinx[1][V_IDX] += 1
                        dsinx[1][COSX_IDX] += 1
                        if not dsinx[1] in self.idx_list:
                            invalid_idx = True
                        constriant_str += 'dsinx[0]*'
                        constriant_str += 'self.mj[self.idx_list.index(dsinx[1])]+'
                    if dcosx[0] != 0:
                        # generator term: -sin(x) * v * df/dcos(x)
                        dcosx[1][V_IDX] += 1
                        dcosx[1][SINX_IDX] += 1
                        if not dcosx[1] in self.idx_list:
                            invalid_idx = True
                        constriant_str += '-dcosx[0]*'
                        constriant_str += 'self.mj[self.idx_list.index(dcosx[1])]+'
                    if dv_2[0] != 0:
                        # generator term: 0.5 * (kc/ms)^2 * d^2f/dv^2
                        if not dv_2[1] in self.idx_list:
                            invalid_idx = True
                        constriant_str += '0.5*dv_2[0]*(self.kc/self.ms)**2*'
                        constriant_str += 'self.mj[self.idx_list.index(dv_2[1])]+'

                    # Remove last '+' sign
                    constriant_str = constriant_str[:-1]
                    constriant_str += '==0'
                    command = 'constraints.append('+constriant_str+')'
                    if not invalid_idx:
                        exec(command)

                print('Num martingale constraints:', len(constraints))
                return constraints


if __name__ == '__main__':

    # System parameters
    state_dim = 5  # State = [x, v, t, sin(x), cos(x)]
    q_polynomial_deg = 4
    T = 5.  # Time horizon

    max_degree = 10
    num_moments = len(generate_unsorted_idx_list(M=max_degree, d=state_dim))
    print('Number of moments:', num_moments)

    idx_list = generate_unsorted_idx_list(M=q_polynomial_deg, d=state_dim)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

    # Safe set:
    # x_lower = -2.5
    # x_upper = 0.0 
    x_lower = -2.5
    x_upper = 0.0

    # Polynomials for safe set:
    # -x^2 - 2.5x >= 0
    pcs_x = np.zeros(len(idx_list)).tolist()
    pcs_x[idx_list.index([2, 0, 0, 0, 0])] = -1.
    pcs_x[idx_list.index([1, 0, 0, 0, 0])] = x_lower

    # -t^2 + Tt >= 0
    pcs_t = np.zeros(len(idx_list)).tolist()
    pcs_t[idx_list.index([0, 0, 2, 0, 0])] = -1.
    pcs_t[idx_list.index([0, 0, 1, 0, 0])] = T

    # -v^2 + 25 >= 0
    pcs_v = np.zeros(len(idx_list)).tolist()
    pcs_v[idx_list.index([0, 2, 0, 0, 0])] = -1.
    pcs_v[idx_list.index([0, 0, 0, 0, 0])] = 25.

    # -sin^2(x) + 1 >= 0
    pcs_sin = np.zeros(len(idx_list)).tolist()
    pcs_sin[idx_list.index([0, 0, 0, 2, 0])] = -1.
    pcs_sin[idx_list.index([0, 0, 0, 0, 0])] = 1.

    # -cos^2(x) + 1 >= 0
    pcs_cos = np.zeros(len(idx_list)).tolist()
    pcs_cos[idx_list.index([0, 0, 0, 0, 2])] = -1.
    pcs_cos[idx_list.index([0, 0, 0, 0, 0])] = 1.

    # Localising matrix polynomials
    pcs_m = [pcs_x, pcs_t]
    pcs_b = []

    # Localising matrix equality constraints
    # t^2x^2 - Lt^2x - Ttx^2 + LTtx >= 0
    idx_list = generate_unsorted_idx_list(M=q_polynomial_deg, d=state_dim)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
    pcs_equality = np.zeros(len(idx_list)).tolist()
    pcs_equality[idx_list.index([2, 0, 2, 0, 0])] = 1.
    pcs_equality[idx_list.index([1, 0, 2, 0, 0])] = -x_lower
    pcs_equality[idx_list.index([2, 0, 1, 0, 0])] = -T
    pcs_equality[idx_list.index([1, 0, 1, 0, 0])] = x_lower*T

    # Localising matrix equality constraints
    # t^2x^2 + 1.3t^2x - 2.3t^2 - 15tx^2 - 19.5tx + 34.5t >= 0
    # idx_list = generate_unsorted_idx_list(M=q_polynomial_deg, d=state_dim)
    # idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
    # pcs_equality = np.zeros(len(idx_list)).tolist()
    # pcs_equality[idx_list.index([2, 0, 2, 0, 0])] = 1.
    # pcs_equality[idx_list.index([1, 0, 2, 0, 0])] = 3.3
    # pcs_equality[idx_list.index([0, 0, 2, 0, 0])] = 2.3
    # pcs_equality[idx_list.index([2, 0, 1, 0, 0])] = -15.
    # pcs_equality[idx_list.index([1, 0, 1, 0, 0])] = -49.5
    # pcs_equality[idx_list.index([0, 0, 1, 0, 0])] = -34.5

    # pcs_equality = np.zeros(len(idx_list)).tolist()
    # pcs_equality[idx_list.index([2, 0, 2, 0, 0])] = 1.
    # pcs_equality[idx_list.index([1, 0, 2, 0, 0])] = 3.3
    # pcs_equality[idx_list.index([0, 0, 2, 0, 0])] = 2.3
    # pcs_equality[idx_list.index([2, 0, 1, 0, 0])] = -2.
    # pcs_equality[idx_list.index([1, 0, 1, 0, 0])] = -6.6
    # pcs_equality[idx_list.index([0, 0, 1, 0, 0])] = -4.6

    # # Localising matrix equality constraints
    # # t^2x^2 + 1.3t^2x - 2.3t^2 - 15tx^2 - 19,5tx + 34.5t >= 0
    # idx_list = generate_unsorted_idx_list(M=q_polynomial_deg, d=state_dim)
    # idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
    # pcs_equality = np.zeros(len(idx_list)).tolist()
    # pcs_equality[idx_list.index([2, 2, 2, 0, 0])] = 1.
    # pcs_equality[idx_list.index([1, 2, 2, 0, 0])] = 3.3
    # pcs_equality[idx_list.index([0, 2, 2, 0, 0])] = 2.3
    # pcs_equality[idx_list.index([2, 0, 2, 0, 0])] = -25.
    # pcs_equality[idx_list.index([1, 0, 2, 0, 0])] = -82.5
    # pcs_equality[idx_list.index([0, 0, 2, 0, 0])] = -57.5
    # pcs_equality[idx_list.index([2, 2, 1, 0, 0])] = -15.
    # pcs_equality[idx_list.index([1, 2, 1, 0, 0])] = -49.5
    # pcs_equality[idx_list.index([0, 2, 1, 0, 0])] = -34.5
    # pcs_equality[idx_list.index([2, 0, 1, 0, 0])] = -375.
    # pcs_equality[idx_list.index([1, 0, 1, 0, 0])] = 1237.5
    # pcs_equality[idx_list.index([0, 0, 1, 0, 0])] = 862.5

    #pcs_b = [pcs_equality, [-i for i in pcs_equality]]
    # print(pcs_b)
    # pcs_b = [pcs_y, [-i if i != 0 else 0 for i in pcs_y]]
    # print(pcs_b)

    # System parameters:
    ks = 5.
    ms = 1.
    kc = 1.

    # Starting state
    starting_state = [-ms*9.81/ks, 0., 1.,
                      math.sin(-ms*9.81/ks), math.cos(-ms*9.81/ks)]

    values = []
    times = []

    scs_scale = 5
    scs_eps = 1.65e-4

    #left 1.65e-4, right 1.75e-4

    for order in range(1):
        ets = escape_time_solver(d=state_dim, M=num_moments, max_degree=max_degree, q_alpha=q_polynomial_deg,
                                 pcs_m=pcs_m, pcs_b=pcs_b, pcs_equality=pcs_equality, y0=starting_state,
                                 max_opt_iters=600000, acc_lookback=5, sys_params=[ks, ms, kc],
                                 scs_eps=scs_eps, scs_scale=scs_scale, solver_verbose=True, exit_time_order=order+1, solver='SCS')

        start_time = time.time()
        values.append(ets.solve(mode='Maximize'))
        runtime = time.time()-start_time
        print('end time:', runtime)
        times.append(runtime)

    #pickle.dump([values, times, max_degree], open('results/bm_mosek_reducedsdp_max.pkl', 'wb'))
