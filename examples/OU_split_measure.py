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
T_IDX = 1


def is_psd(x):
    return np.all(np.linalg.eigvals(x) >= 0)


class escape_time_solver:

    def __init__(self, a, T, d, max_degree, pcs_m, pcs_b1, pcs_b2, y0, scs_scale,
                 q_alpha_m, q_alpha_b1, q_alpha_b2, max_opt_iters=5000,
                 acc_lookback=5, solver_verbose=True, exit_time_order=1, solver='SCS'):
        self.d = d  # Dimension of system
        self.a = a  # Exit level
        self.T = T  # Time horizon
        self.exit_time_order = exit_time_order
        self.solver = solver
        self.scs_scale = scs_scale
        self.max_degree = max_degree

        # Generate list of moment indexes
        idx_list_m = generate_unsorted_idx_list(M=max_degree, d=d)
        self.idx_list_m = sorted(idx_list_m, key=cmp_to_key(index_comparison))
        idx_list_b = generate_unsorted_idx_list(M=max_degree, d=1)
        self.idx_list_b = sorted(idx_list_b, key=cmp_to_key(index_comparison))

        # Moment arrays
        self.mj = [cp.Variable() for i in range(len(self.idx_list_m))]
        self.bj1 = [cp.Variable() for i in range(len(self.idx_list_b))]
        self.bj2 = [cp.Variable() for i in range(len(self.idx_list_b))]

        # Moment matrices
        self.moment_matrix_mj = None
        self.moment_matrix_bj1 = None
        self.moment_matrix_bj2 = None

        # Localizing matrices
        self.localising_matrices_mj = None
        self.localising_matrices_bj1 = None
        self.localising_matrices_bj2 = None
        self.q_alpha_m = q_alpha_m
        self.q_alpha_b1 = q_alpha_b1
        self.q_alpha_b2 = q_alpha_b2

        # Moment idx dictionary (Used for creating LMs)
        # Key: Str(i_j)
        # Eg:  entry 1,3 -> '1_3'
        # Value: moment index (multi-index)
        self.mm_dict_mj = {}
        self.mm_dict_bj1 = {}
        self.mm_dict_bj2 = {}

        # Highest degree of monomials for polynomial q
        # self.q_alpha = q_alpha
        # self.equality_poly_degree = q_alpha_equality

        # Polynomials used to define localizing matrices
        self.polynomial_coeffs_m = pcs_m
        self.polynomial_coeffs_b1 = pcs_b1
        self.polynomial_coeffs_b2 = pcs_b2

        self.y0 = y0

        # Solver parameters
        self.solver_verbose = solver_verbose
        self.max_opt_iters = max_opt_iters
        self.acc_lookback = acc_lookback

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
        print('Number of localising matrices (BJ1):',
              len(self.localising_matrices_bj1))
        print('Number of localising matrices (BJ2):',
              len(self.localising_matrices_bj2))
        print('Time to create localising matrix:',
              time.time()-start_time, '\n')

        start_time = time.time()
        constraints += self.lm_constraints()
        print('Time to create lm constraints:', time.time()-start_time, '\n')

        start_time = time.time()
        constraints += self.martingale_constraints()
        print('Time to create martingale constraints:',
              time.time()-start_time, '\n')

        print('Number of total constraints: ', len(constraints), '\n')

        start_time = time.time()
        obj = cp.Maximize(self.exit_time_order*self.mj[self.idx_list_m.index([0, self.exit_time_order-1])]) if mode == 'Maximize' else cp.Minimize(
            self.exit_time_order*self.mj[self.idx_list_m.index([0, self.exit_time_order-1])])
        print('Objective', obj)
        print('Time to create objective:', time.time()-start_time)

        start_time = time.time()
        self.prob = cp.Problem(obj, constraints)
        print('Time to create problem object:', time.time()-start_time)

        if self.solver == 'SCS':
            self.prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.SCS,
                            acceleration_lookback=self.acc_lookback, scale=self.scs_scale)
        elif self.solver == 'GUROBI':
            self.prob.solve(verbose=self.solver_verbose, solver=cp.GUROBI)
        else:
            self.prob.solve(verbose=self.solver_verbose, solver=cp.MOSEK)

        print('MJs')
        for mj in self.mj[:5]:
            print(mj.value)
        print('\nBJ1s')
        for bj in self.bj1[:5]:
            print(bj.value)
        print('\nBJ2s')
        for bj in self.bj2[:5]:
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
        mm_degree = self.max_degree//2

        # Generate moment matrix for occupation measure
        idx_list = generate_unsorted_idx_list(M=mm_degree, d=self.d)
        self.moment_matrix_mj = cp.Variable(
            (len(idx_list), len(idx_list)), PSD=True)

        # Generate moment matrix for exit measures
        idx_list = generate_unsorted_idx_list(M=mm_degree, d=1)
        self.moment_matrix_bj1 = cp.Variable(
            (len(idx_list), len(idx_list)), PSD=True)
        self.moment_matrix_bj2 = cp.Variable(
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
        self.localising_matrices_bj1 = []
        self.localising_matrices_bj2 = []

        # Create empty localizing matrices for MJs
        num_constraint_polys_mj = len(self.polynomial_coeffs_m)
        for j in range(num_constraint_polys_mj):
            # Allocate space for local matrix
            lm_dim = len(generate_unsorted_idx_list(
                M=(self.max_degree//2)-1, d=self.d))
            self.localising_matrices_mj.append(
                cp.Variable((lm_dim, lm_dim), PSD=True))

        # Create empty localizing matrices for BJs
        num_constraint_polys_bj1 = len(self.polynomial_coeffs_b1)
        for j in range(num_constraint_polys_bj1):
            # Allocate space for local matrix
            lm_dim = len(generate_unsorted_idx_list(
                M=(self.max_degree//2)-1, d=1))
            self.localising_matrices_bj1.append(
                cp.Variable((lm_dim, lm_dim), PSD=True))
            self.localising_matrices_bj2.append(
                cp.Variable((lm_dim, lm_dim), PSD=True))

    # Assign moment matrix entries to moments
    # -----------------------------------------------------------------
    # Inputs:
    # 	None
    # Returns:
    #  	List of constraints that assign moments to moment matrices

    def mm_constraints(self):
        constraints = []

        vis_mm_mj = {}
        vis_mm_bj = {}

        for i in range(self.moment_matrix_mj.shape[0]):
            for j in range(self.moment_matrix_mj.shape[1]):
                moment_idx = [sum(x) for x in zip(
                    self.idx_list_m[i], self.idx_list_m[j])]

                constraints.append(self.moment_matrix_mj[i][j]
                                   == self.mj[self.idx_list_m.index(moment_idx)])

                # Create dictionary entry for LMs
                self.mm_dict_mj[str(i)+'_'+str(j)] = moment_idx

                # Add to dictionary for visualisation
                vis_mm_mj[(i, j)] = moment_idx

        for i in range(self.moment_matrix_bj1.shape[0]):
            for j in range(self.moment_matrix_bj1.shape[1]):
                moment_idx = [sum(x) for x in zip(
                    self.idx_list_b[i], self.idx_list_b[j])]

                constraints.append(self.moment_matrix_bj1[i][j]
                                   == self.bj1[self.idx_list_b.index(moment_idx)])
                constraints.append(self.moment_matrix_bj2[i][j]
                                   == self.bj2[self.idx_list_b.index(moment_idx)])

                # Create dictionary entry for LMs
                self.mm_dict_bj1[str(i)+'_'+str(j)] = moment_idx
                self.mm_dict_bj2[str(i)+'_'+str(j)] = moment_idx

                # Add to dictionary for visualisation
                vis_mm_bj[(i, j)] = moment_idx

        # Visualise moment matrix
        mm = [[None for _ in range(self.moment_matrix_mj.shape[1])]
              for _ in range(self.moment_matrix_mj.shape[0])]
        for key in vis_mm_mj:
            i = key[0]
            j = key[1]
            entry = vis_mm_mj[key]
            mm[i][j] = entry
        print('\nMoment matrix (occupation measures):')
        for row in mm:
            print(row)
        print(' ')

        mm = [[None for _ in range(self.moment_matrix_bj1.shape[1])]
              for _ in range(self.moment_matrix_bj1.shape[0])]
        for key in vis_mm_bj:
            i = key[0]
            j = key[1]
            entry = vis_mm_bj[key]
            mm[i][j] = entry
        print('\nMoment matrix (Exit measures):')
        for row in mm:
            print(row)
        print(' ')

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
                    mm_idx = self.mm_dict_mj[str(i)+'_'+str(j)]

                    # list of alpha indexes
                    idx_list = generate_unsorted_idx_list(
                        M=self.q_alpha_m, d=self.d)
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
                                str(self.idx_list_m.index(lm_idx))+']+'

                    # Remove ending '+'
                    constriant_str_mj = constriant_str_mj[:-1]
                    # Append constraint to list and execute
                    command_mj = 'constraints.append('+constriant_str_mj+')'
                    exec(command_mj)

        # Obtain the number of localising matrices for BJ1
        num_lm_bj1 = len(self.localising_matrices_bj1)
        for cur_lm in range(num_lm_bj1):
            # Link to moments constraints
            for i in range(self.localising_matrices_bj1[cur_lm].shape[0]):
                for j in range(self.localising_matrices_bj1[cur_lm].shape[1]):
                    # Get moment index from moment matrix
                    mm_idx = self.mm_dict_bj1[str(i)+'_'+str(j)]

                    # list of alpha indexes
                    idx_list = generate_unsorted_idx_list(
                        M=self.q_alpha_b1, d=1)
                    idx_list = sorted(
                        idx_list, key=cmp_to_key(index_comparison))

                    # Build constraint instruction as a string
                    constriant_str_bj = ''
                    constriant_str_bj += 'self.localising_matrices_bj1['+str(
                        cur_lm)+']['+str(i)+']['+str(j)+']=='

                    for alpha in range(len(idx_list)):
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
                        if self.polynomial_coeffs_b1[cur_lm][alpha] != 0:
                            # Generate constraint string
                            constriant_str_bj += 'self.polynomial_coeffs_b1[cur_lm]['+str(
                                alpha)+']*'
                            constriant_str_bj += 'self.bj1[' + \
                                str(self.idx_list_b.index(lm_idx))+']+'

                    # Remove ending '+'
                    constriant_str_bj = constriant_str_bj[:-1]
                    # Append constraint to list and execute
                    command_bj = 'constraints.append('+constriant_str_bj+')'
                    exec(command_bj)

        # Obtain the number of localising matrices for BJ2
        num_lm_bj2 = len(self.localising_matrices_bj2)
        for cur_lm in range(num_lm_bj2):
            # Link to moments constraints
            for i in range(self.localising_matrices_bj2[cur_lm].shape[0]):
                for j in range(self.localising_matrices_bj2[cur_lm].shape[1]):
                    # Get moment index from moment matrix
                    mm_idx = self.mm_dict_bj2[str(i)+'_'+str(j)]

                    # list of alpha indexes
                    idx_list = generate_unsorted_idx_list(
                        M=self.q_alpha_b2, d=1)
                    idx_list = sorted(
                        idx_list, key=cmp_to_key(index_comparison))

                    # Build constraint instruction as a string
                    constriant_str_bj = ''
                    constriant_str_bj += 'self.localising_matrices_bj2['+str(
                        cur_lm)+']['+str(i)+']['+str(j)+']=='

                    for alpha in range(len(idx_list)):
                        # Get moment index by summing alpha index with moment matrix index
                        lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
                        if self.polynomial_coeffs_b2[cur_lm][alpha] != 0:
                            # Generate constraint string
                            constriant_str_bj += 'self.polynomial_coeffs_b2[cur_lm]['+str(
                                alpha)+']*'
                            constriant_str_bj += 'self.bj2[' + \
                                str(self.idx_list_b.index(lm_idx))+']+'

                    # Remove ending '+'
                    constriant_str_bj = constriant_str_bj[:-1]
                    # Append constraint to list and execute
                    command_bj = 'constraints.append('+constriant_str_bj+')'
                    exec(command_bj)

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
        constraints.append(-self.bj1[0] - self.bj2[0] + 1 == 0)
        #constraints.append(self.mj[0] >= 0)

        ################ Degree 1 #########################################################
        start = next(i for i, v in enumerate(self.idx_list_m)
                     if np.sum(self.idx_list_m[i]) == 1)
        end = next(i for i, v in enumerate(self.idx_list_m)
                   if np.sum(self.idx_list_m[i]) == 2)

        for n in range(start, end):
            invalid_idx = False
            f = copy.deepcopy(self.idx_list_m[n])  # Get the test function

            # For each test function perform the following derivatives:
            # dx, dt
            dx = mono_derivative(f, [0])
            dt = mono_derivative(f, [1])

            constriant_str = ''

            # Bj constraint
            time_degree = f[T_IDX]
            x_degree = f[X_IDX]
            constriant_str += '-1. * self.a**x_degree * self.bj1[self.idx_list_b.index([time_degree])]+'
            constriant_str += '-1. * self.T**time_degree * self.bj2[self.idx_list_b.index([x_degree])]+'

            # y0 constraint: f(y0)
            y0_const = 1.0
            for i in range(len(f)):
                if f[i] != 0:
                    y0_const *= self.y0[i]**f[i]
            constriant_str += 'y0_const+'

            # Mj constraints
            if dt[0] != 0:
                # generator term: df/dt
                constriant_str += '0.01*dt[0]*'
                constriant_str += 'self.mj[self.idx_list_m.index(dt[1])]+'
            if dx[0] != 0:
                # generator term: -x * df/dx
                dx[1][X_IDX] += 1  # Increment degree of monomial
                if not dx[1] in self.idx_list_m:
                    invalid_idx = True
                constriant_str += '-1.*dx[0]*'
                constriant_str += 'self.mj[self.idx_list_m.index(dx[1])]+'

            constriant_str = constriant_str[:-1] 	# Remove last '+' sign
            constriant_str += '==0'
            command = 'constraints.append('+constriant_str+')'

            if not invalid_idx:
                exec(command)

        # ################ Degree 2 and above ###############################################
        # Get position in index list where monomial degree is 2
        start = next(i for i, v in enumerate(self.idx_list_m)
                     if np.sum(self.idx_list_m[i]) == 2)

        # n represents the index of the monomial test function
        for n in range(start, len(self.mj)):
            invalid_idx = False
            f = copy.deepcopy(self.idx_list_m[n])  # Get the test function

            # For each test function perform the following derivatives:
            # dx, dt, dx^2
            dx = mono_derivative(f, [0])
            dt = mono_derivative(f, [1])
            dx_2 = mono_derivative(f, [0, 0])

            constriant_str = ''

            # Bj constraint
            time_degree = f[T_IDX]
            x_degree = f[X_IDX]
            constriant_str += '-1. * self.a**x_degree * self.bj1[self.idx_list_b.index([time_degree])]+'
            constriant_str += '-1. * self.T**time_degree * self.bj2[self.idx_list_b.index([x_degree])]+'

            # y0 constraint: f(y0)
            y0_const = 1.0
            for i in range(len(f)):
                if f[i] != 0:
                    y0_const *= self.y0[i]**f[i]
            constriant_str += 'y0_const+'

            # Mj constraints
            if dt[0] != 0:
                # generator term: df/dt
                constriant_str += '0.01*dt[0]*'
                constriant_str += 'self.mj[self.idx_list_m.index(dt[1])]+'
            if dx[0] != 0:
                # generator term: -x * df/dx
                dx[1][X_IDX] += 1  # Increment degree of monomial
                if not dx[1] in self.idx_list_m:
                    invalid_idx = True
                constriant_str += '-1.*dx[0]*'
                constriant_str += 'self.mj[self.idx_list_m.index(dx[1])]+'
            if dx_2[0] != 0:
                # generator term: d^2f/dx^2
                if not dx_2[1] in self.idx_list_m:
                    invalid_idx = True
                constriant_str += 'dx_2[0]*'
                constriant_str += 'self.mj[self.idx_list_m.index(dx_2[1])]+'

            constriant_str = constriant_str[:-1] 	# Remove last '+' sign
            constriant_str += '==0'
            command = 'constraints.append('+constriant_str+')'
            if not invalid_idx:
                exec(command)

        print('Num martingale constraints:', len(constraints))
        return constraints


if __name__ == '__main__':

    # System parameters
    state_dim = 2  # State = [y, t]
    q_alpha_m = 2
    q_alpha_b1 = 2
    q_alpha_b2 = 1
    T = 2.  # Time horizon
    a = 2.   # Exit level of safe set

    max_degree = 10
    num_moments = len(generate_unsorted_idx_list(M=max_degree, d=2))
    print('Number of moments:', num_moments)

    idx_list = generate_unsorted_idx_list(M=q_alpha_m, d=state_dim)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

    # Polynomials for safe set:
    # -x + 2 >= 0
    pcs_x = np.zeros(len(idx_list)).tolist()
    pcs_x[idx_list.index([1, 0])] = -1.
    pcs_x[idx_list.index([0, 0])] = 2.

    # -t^2 + Tt >= 0
    pcs_t = np.zeros(len(idx_list)).tolist()
    pcs_t[idx_list.index([0, 2])] = -1.
    pcs_t[idx_list.index([0, 1])] = T

    idx_list = generate_unsorted_idx_list(M=q_alpha_b1, d=1)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
    pcs_b1 = np.zeros(len(idx_list)).tolist()
    pcs_b1[idx_list.index([2])] = -1.
    pcs_b1[idx_list.index([1])] = T

    idx_list = generate_unsorted_idx_list(M=q_alpha_b2, d=1)
    idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
    pcs_b2 = np.zeros(len(idx_list)).tolist()
    pcs_b2[idx_list.index([1])] = -1.
    pcs_b2[idx_list.index([0])] = a

    # Localising matrix polynomials
    pcs_m = [pcs_x, pcs_t]

    # Starting state [y=0.0, t=0]
    starting_state = [0., 0.]

    scs_scale = 100

    values = []
    times = []

    for order in range(1):
        ets = escape_time_solver(a=a, T=T, d=state_dim, max_degree=max_degree, q_alpha_m=q_alpha_m,
                                 q_alpha_b1=q_alpha_b1, q_alpha_b2=q_alpha_b2,
                                 pcs_m=pcs_m, pcs_b1=[pcs_b1], pcs_b2=[pcs_b2],
                                 y0=starting_state, scs_scale=scs_scale,
                                 max_opt_iters=2000000, acc_lookback=10,
                                 solver_verbose=True, exit_time_order=order+2, solver='MOSEK')

        start_time = time.time()
        values.append(ets.solve(mode='Minimize'))
        runtime = time.time()-start_time
        print('end time:', runtime)
        times.append(runtime)

    #pickle.dump([values, times, max_degree], open('results/OU_max_M'+str(max_degree)+'.pkl', 'wb'))
