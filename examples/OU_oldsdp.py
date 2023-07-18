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
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:

	def __init__(self, d, M, max_degree, pcs_m, pcs_b_inner, pcs_b_outer, sys_params, q_alpha=5, y0=0.2, lm_vis=True, max_opt_iters=5000,
			     acc_lookback=5, warm_start=True, solver_verbose=True, exit_time_order=1, solver='SCS'):
		self.d = d # Dimension of system 
		self.M = M # Number of moments (multi-indexes)
		self.exit_time_order = exit_time_order
		self.solver = solver

		# Generate list of moment indexes
		idx_list = generate_unsorted_idx_list(M=max_degree,d=d)
		self.idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
		self.idx_list = self.idx_list[:M]

		# Moment arrays
		# self.mj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		# self.bj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		self.mj = [cp.Variable() for i in range(len(self.idx_list))]
		self.bj = [cp.Variable() for i in range(len(self.idx_list))]

		# Moment matrices
		self.mm_mj = None
		self.mm_bj = None

		# Localizing matrices
		self.lm_mj = None
		self.lm_bj = None
		self.lm_bj_inner = None
		self.lm_bj_outer = None
		#self.lm_bj2 = None

		# Moment idx dictionary (Used for creating LMs)
		self.mm_dict = {}	# Key: Str(moment matrix idx,_,i,_,j)
							# Eg:  Moment matrix 2, entry 1,3 -> 2_1_3
							# Value: moment index (multi-index)

		# Dictionary to keep track of how many localizing matrices per constraint polynomial
		self.lm_count_dict_mj = {}		# Key: constraint polynomial idx
		self.lm_count_dict_bj = {}		# Value: number of localizing matrices


		# Highest degree of monomials for polynomial q
		self.q_alpha = q_alpha

		# Polynomials used to define localizing matrices
		self.polynomial_coeffs_m = pcs_m
		self.polynomial_coeffs_bj_inner = pcs_b_inner
		self.polynomial_coeffs_bj_outer = pcs_b_outer

		self.y0 = y0

		# Visualize localizing matrices
		self.lm_vis = lm_vis
		self.max_opt_iters = max_opt_iters
		self.acc_lookback = acc_lookback

		# Solver parameters
		self.warm_start = warm_start
		self.solver_verbose = solver_verbose

		# System parameters
		self.ks = sys_params[0]
		self.ms = sys_params[1]
		self.kc = sys_params[2]
		self.g = 9.81

		# Convex opt problem
		self.prob = None



	def solve(self, mode='Maximize'):
		constraints = []

		start_time = time.time()
		self.create_moment_matrix()
		print('Number of moment matrices:', len(self.mm_mj))
		print('Time to create moment matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.mm_constraints()
		print('Time to create mm constraints:', time.time()-start_time)

		start_time = time.time()
		self.create_local_matrix()
		print('Number of localising matrices (MJ):', len(self.lm_mj[0]))
		print('Number of localising matrices (BJ Inner):', len(self.lm_bj_inner))
		print('Number of localising matrices (BJ Outer):', len(self.lm_bj_outer))
		print('Time to create localising matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.lm_constraints()
		print('Time to create lm constraints:', time.time()-start_time)

		start_time = time.time()
		constraints += self.martingale_constraints()
		print('Time to create martingale constraints:', time.time()-start_time)


		#self.idx_list.index(dt[1])
		start_time = time.time()
		obj = cp.Maximize(self.exit_time_order*self.mj[self.idx_list.index([0, self.exit_time_order-1])]) if mode == 'Maximize' else cp.Minimize(
            self.exit_time_order*self.mj[self.idx_list.index([0, self.exit_time_order-1])])
		print('Objective', obj)
		print('Time to create objective:', time.time()-start_time)

		start_time = time.time()
		self.prob = cp.Problem(obj, constraints)
		print('Time to create problem object:', time.time()-start_time)

		
		if self.solver == 'SCS':
			self.prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.SCS, 
			       			acceleration_lookback=self.acc_lookback)
		elif self.solver == 'GUROBI':
			self.prob.solve(verbose=self.solver_verbose,solver=cp.GUROBI)
		else:
			self.prob.solve(verbose=self.solver_verbose,solver=cp.MOSEK)

		print('MJs')
		for mj in self.mj[:5]:
			print(mj.value)
		print('\nBJs')
		for bj in self.bj[:5]:
			print(bj.value)

		print(' ')

		print(self.prob.value)
		print(self.prob.status)




	def num_mms(self):
		for mm_idx in range(self.M):
			idx_list = generate_unsorted_idx_list(M=mm_idx,d=self.d)
			idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			if len(idx_list) > len(self.idx_list):
				return mm_idx
			for i in range(len(idx_list)):
				for j in range(len(idx_list)):
					moment_idx = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]
					if moment_idx not in self.idx_list:
						return mm_idx
		return mm_idx+1
	def create_moment_matrix(self):
		self.mm_mj = []
		self.mm_bj = []

		max_num_mms = self.num_mms()

		# Instantiate the moment matrices 
		for i in range(max_num_mms):

			# Generate list of moment indexes for current moment matrix
			idx_list = generate_unsorted_idx_list(M=i,d=self.d)
			idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			
			# Allocate space for moment matrix
			self.mm_mj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))
			self.mm_bj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))




	def is_lm_valid(self, q_poly_multi_idx, lm_dim):
		for i in range(lm_dim):
			for j in range(lm_dim):
				mm_moment = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]
				lm_entry = [sum(x) for x in zip(mm_moment, q_poly_multi_idx)]
				if not lm_entry in self.idx_list:
					return False
		return True
	def create_local_matrix(self):
		self.lm_mj = []
		self.lm_bj_inner = []
		self.lm_bj_outer = []

		# Create empty localizing matrices for MJs
		num_constraint_polys_mj = len(self.polynomial_coeffs_m)
		for j in range(num_constraint_polys_mj):
			self.lm_mj.append([])
			# Calculate the max number of localizing matrices
			for i in range(len(self.mm_mj)):

				# Generate list of moment indexes for q alpha
				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
				# Find the largest index in the q polynomial coeff that isn't 0
				for k in range(len(idx_list)-1,-1,-1):
					if self.polynomial_coeffs_m[j][k] != 0:
						break

				# Check to see if this lm contains valid indexes
				if self.is_lm_valid(idx_list[k], self.mm_mj[i].shape[0]):
					# Allocate space for local matrix
					self.lm_mj[-1].append(cp.Variable((self.mm_mj[i].shape[0],self.mm_mj[i].shape[1]), PSD=True))
				else:
					break
		


		# Create empty localizing matrices for BJs
		num_constraint_polys_bj = len(self.polynomial_coeffs_bj_inner)
		for j in range(num_constraint_polys_bj):
			self.lm_bj_inner.append([])
			self.lm_bj_outer.append([])
			# Calculate the max number of localizing matrices
			for i in range(len(self.mm_bj)):

				# Generate list of moment indexes for q alpha
				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
				# Find the largest index in the q polynomial coeff that isn't 0
				for k in range(len(idx_list)-1,-1,-1):
					if self.polynomial_coeffs_bj_inner[j][k] != 0:
						break

				# Check to see if this lm contains valid indexes
				if self.is_lm_valid(idx_list[k], self.mm_bj[i].shape[0]):
					# Allocate space for local matrix
					self.lm_bj_inner[-1].append(cp.Variable((self.mm_bj[i].shape[0],self.mm_bj[i].shape[1]), PSD=True))
					self.lm_bj_outer[-1].append(cp.Variable((self.mm_bj[i].shape[0],self.mm_bj[i].shape[1]), PSD=True))
				else:
					break
	

					
	def mm_constraints(self):
		constraints = []

		for mm_idx in range(len(self.mm_mj)):
			for i in range(self.mm_mj[mm_idx].shape[0]):
				for j in range(self.mm_mj[mm_idx].shape[1]):

					moment_idx = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]

					constraints.append(self.mm_mj[mm_idx][i][j]
						               == self.mj[self.idx_list.index(moment_idx)])
					constraints.append(self.mm_bj[mm_idx][i][j]
						               == self.bj[self.idx_list.index(moment_idx)])	
					
					# Create dictionary entry for LMs
					self.mm_dict[str(mm_idx)+'_'+str(i)+'_'+str(j)] = moment_idx

		return constraints



	def lm_constraints(self):
		constraints = []

		# Obtain the contraints for the MJ localizing matrices
		num_constraint_polys_mj = len(self.polynomial_coeffs_m)
		# Calculate the number of localising matrices per constraint polynomial
		num_lms_mj = int(len(self.lm_mj)/num_constraint_polys_mj)


		for cur_constraint_poly in range(num_constraint_polys_mj):
			# Calculate the number of localising matrices for the current constraint polynomial
			num_lms_mj = len(self.lm_mj[cur_constraint_poly])
			for cur_mat in range(num_lms_mj):
				# Link to moments constraints
				for i in range(self.lm_mj[cur_constraint_poly][cur_mat].shape[0]):
					for j in range(self.lm_mj[cur_constraint_poly][cur_mat].shape[1]):
						
						# Get index from corresponding moment matrix
						mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

						# list of alpha indexes
						idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
						idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

						# Build constraint instruction as a string
						constriant_str_mj = ''
						constriant_str_mj += 'self.lm_mj['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='

						for alpha in range(len(idx_list)):

							# Get moment index by summing alpha index with moment matrix index
							lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

							if self.polynomial_coeffs_m[cur_constraint_poly][alpha] != 0:
								# Generate constraint string 
								constriant_str_mj += 'self.polynomial_coeffs_m[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_mj += 'self.mj['+str(self.idx_list.index(lm_idx))+']+'

						# Remove ending '+' 
						constriant_str_mj = constriant_str_mj[:-1]
	
						# Append constraint to list and execute 
						command_mj = 'constraints.append('+constriant_str_mj+')'

						exec(command_mj)


		# Obtain the contraints for the BJ (Inner) localizing matrices
		num_constraint_polys_bj_inner = len(self.polynomial_coeffs_bj_inner)		#Inner and Outer are negatives of each other so lengths are the same
		# Calculate the number of localising matrices per constraint polynomial

		for cur_constraint_poly in range(num_constraint_polys_bj_inner):
			num_lms_bj_inner = len(self.lm_bj_inner[cur_constraint_poly])
			for cur_mat in range(num_lms_bj_inner):
				# Link to moments constraints
				for i in range(self.lm_bj_inner[cur_constraint_poly][cur_mat].shape[0]):
					for j in range(self.lm_bj_inner[cur_constraint_poly][cur_mat].shape[1]):
						
						# Get index from corresponding moment matrix
						mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

						# list of alpha indexes
						idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
						idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

						# Build constraint instruction as a string
						constriant_str_bj_inner = ''
						constriant_str_bj_inner += 'self.lm_bj_inner['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='
						
						for alpha in range(len(idx_list)):

							# Get moment index by summing alpha index with moment matrix index
							lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

							if self.polynomial_coeffs_bj_inner[cur_constraint_poly][alpha] != 0:
								constriant_str_bj_inner += 'self.polynomial_coeffs_bj_inner[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_bj_inner += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'
							
						# Remove ending '+' 
						constriant_str_bj_inner = constriant_str_bj_inner[:-1]
						
						# Append constraint to list and execute 
						command_bj_inner = 'constraints.append('+constriant_str_bj_inner+')'

						exec(command_bj_inner)
		



		# Obtain the contraints for the BJ (Outer) localizing matrices
		num_constraint_polys_bj_outer = len(self.polynomial_coeffs_bj_outer)		#Inner and Outer are negatives of each other so lengths are the same
		# Calculate the number of localising matrices per constraint polynomial

		for cur_constraint_poly in range(num_constraint_polys_bj_outer):
			num_lms_bj_outer = len(self.lm_bj_outer[cur_constraint_poly])
			for cur_mat in range(num_lms_bj_outer):
				# Link to moments constraints
				for i in range(self.lm_bj_outer[cur_constraint_poly][cur_mat].shape[0]):
					for j in range(self.lm_bj_outer[cur_constraint_poly][cur_mat].shape[1]):
						
						# Get index from corresponding moment matrix
						mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

						# list of alpha indexes
						idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
						idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

						# Build constraint instruction as a string
						constriant_str_bj_outer = ''
						constriant_str_bj_outer += 'self.lm_bj_outer['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='

						for alpha in range(len(idx_list)):

							# Get moment index by summing alpha index with moment matrix index
							lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

							if self.polynomial_coeffs_bj_outer[cur_constraint_poly][alpha] != 0:
								constriant_str_bj_outer += 'self.polynomial_coeffs_bj_outer[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_bj_outer += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'

						# Remove ending '+' 
						constriant_str_bj_outer = constriant_str_bj_outer[:-1]
						
						# Append constraint to list and execute 
						command_bj_outer = 'constraints.append('+constriant_str_bj_outer+')'

						exec(command_bj_outer)

		return constraints




	# Martingale constraints
	# states: [x, y, gamma (g), t, cos(g), sin(g)]
	# *Refer to RN notebook 12/4 for generator  
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
			# dx, dt
			dx = mono_derivative(f, [0])
			dt = mono_derivative(f, [1])

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
			if dt[0] != 0:
				# generator term: df/dt
				constriant_str += 'dt[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dt[1])]+'
			if dx[0] != 0:
				# generator term: -alpha * x * df/dx
				dx[1][X_IDX] += 1	# Increment degree of monomial
				if not dx[1] in self.idx_list:
					invalid_idx = True
				constriant_str += '-0.525*dx[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'

			if not invalid_idx:
				exec(command)

		# ################ Degree 2 and above ###############################################
		# Get position in index list where monomial degree is 2
		start = next(i for i, v in enumerate(self.idx_list)
						if np.sum(self.idx_list[i]) == 2)

		# n represents the index of the monomial test function
		for n in range(start, len(self.mj)):
			invalid_idx = False
			f = copy.deepcopy(self.idx_list[n])  # Get the test function

			# For each test function perform the following derivatives:
			# dx, dt, dx^2
			dx = mono_derivative(f, [0])
			dt = mono_derivative(f, [1])
			dx_2 = mono_derivative(f, [0, 0])

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
			if dt[0] != 0:
				# generator term: df/dt
				constriant_str += 'dt[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dt[1])]+'
			if dx[0] != 0:
				# generator term: -alpha * x * df/dx
				dx[1][X_IDX] += 1	# Increment degree of monomial
				if not dx[1] in self.idx_list:
					invalid_idx = True
				constriant_str += '-0.525*dx[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'
			if dx_2[0] != 0:
				# generator term: 0.5 * sigma^2 * d^2f/dx^2
				if not dx_2[1] in self.idx_list:
					invalid_idx = True
				constriant_str += '0.5*(0.688**2)*dx_2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx_2[1])]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			if not invalid_idx:
				exec(command)

		print('Num martingale constraints:', len(constraints))
		return constraints




if __name__ == '__main__':


	state_dim = 2  # State = [y, t]
	q_polynomial_deg = 2
	T = 30.  # Time horizon

	max_degree = 8

	idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

	# Polynomials for safe set:
    # -x^2 + 11x - 10 >= 0
	pcs_x = np.zeros(len(idx_list)).tolist()
	pcs_x[idx_list.index([2, 0])] = -1.
	pcs_x[idx_list.index([1, 0])] = 11.
	pcs_x[idx_list.index([0, 0])] = -10.

	# -t^2 + Tt >= 0
	pcs_t = np.zeros(len(idx_list)).tolist()
	pcs_t[idx_list.index([0, 2])] = -1.
	pcs_t[idx_list.index([0, 1])] = T

	# Localising matrix polynomials
	pcs_m = [pcs_x, pcs_t]

	pcs_b_inner = [[-i if i != 0 else 0 for i in pcs_x]]
	pcs_b_outer = [pcs_x]

	num_moments = len(generate_unsorted_idx_list(M=max_degree,d=5))

	# System parameters:
	ks = 5
	ms = 1
	kc = 0.5
	ks = 5.
	ms = 1.
	kc = 1.
	# state = [position, velocity, time]
	starting_state = [9.6, 0.]

	ets = escape_time_solver(d=state_dim, M=num_moments, max_degree=max_degree, q_alpha=q_polynomial_deg, 
		                     pcs_m=pcs_m, pcs_b_inner=pcs_b_inner, pcs_b_outer=pcs_b_outer,
							 lm_vis=False, y0=starting_state,
		                     max_opt_iters=100000, acc_lookback=10, sys_params=[ks,ms,kc],
		                     solver_verbose=True, exit_time_order=1, solver='SCS', 
							 warm_start=False)

	start_time = time.time()
	ets.solve(mode='Maximize')
	print('end time:',time.time()-start_time)

	#pickle.dump(ets, open('results/9_28.pkl', 'wb'))
	
	#pickle.dump(ets, open('results/SMD/time_space_SS/M265_2000000_acc5_min_deg1.pkl', 'wb'))
	