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
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:

	def __init__(self, d, M, pcs_m, pcs_b, sys_params, y0, q_alpha=5, max_opt_iters=5000,
			     acc_lookback=5, warm_start=True, solver_verbose=True, exit_time_order=1, solver='SCS'):
		self.d = d # Dimension of system 
		self.M = M # Number of moments (multi-indexes)
		self.exit_time_order = exit_time_order
		self.solver = solver

		# Generate list of moment indexes
		idx_list = generate_unsorted_idx_list(M=20,d=d)
		self.idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
		self.idx_list = self.idx_list[:M]

		# Moment arrays
		# self.mj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		# self.bj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		self.mj = [cp.Variable() for i in range(len(self.idx_list))]
		self.bj = [cp.Variable() for i in range(len(self.idx_list))]

		# Moment matrices
		self.moment_matrix_mj = None
		self.moment_matrix_bj = None

		# Localizing matrices
		self.localising_matrices_mj = None
		self.localising_matrices_mj = None

		# Moment idx dictionary (Used for creating LMs)
		self.mm_dict = {}	# Key: Str(i_j)
							# Eg:  entry 1,3 -> '1_3'
							# Value: moment index (multi-index)

		# Highest degree of monomials for polynomial q
		self.q_alpha = q_alpha

		# Polynomials used to define localizing matrices
		self.polynomial_coeffs_m = pcs_m
		self.polynomial_coeffs_b = pcs_b

		self.y0 = y0

		# Solver parameters
		self.warm_start = warm_start
		self.solver_verbose = solver_verbose
		self.max_opt_iters = max_opt_iters
		self.acc_lookback = acc_lookback

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
		print('Time to create moment matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.mm_constraints()
		print('Time to create mm constraints:', time.time()-start_time)

		start_time = time.time()
		self.create_local_matrix()
		print('Number of localising matrices (MJ):', len(self.localising_matrices_mj))
		print('Number of localising matrices (BJ):', len(self.localising_matrices_bj))
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
		# self.mm_mj = []
		# self.mm_bj = []

		# max_num_mms = self.num_mms()

		# # Instantiate the moment matrices 
		# for i in range(max_num_mms):

		# 	# Generate list of moment indexes for current moment matrix
		# 	idx_list = generate_unsorted_idx_list(M=i,d=self.d)
		# 	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			
		# 	# Allocate space for moment matrix
		# 	self.mm_mj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))
		# 	self.mm_bj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))

		mm_degree = self.num_mms()-1
		idx_list = generate_unsorted_idx_list(M=mm_degree, d=self.d)
		self.moment_matrix_mj = cp.Variable((len(idx_list),len(idx_list)), PSD=True)
		self.moment_matrix_bj = cp.Variable((len(idx_list),len(idx_list)), PSD=True)


	def is_lm_valid(self, q_poly_multi_idx, lm_dim):
		for i in range(lm_dim):
			for j in range(lm_dim):
				mm_moment = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]
				lm_entry = [sum(x) for x in zip(mm_moment, q_poly_multi_idx)]
				if not lm_entry in self.idx_list:
					return False
		return True
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
			for k in range(len(idx_list)-1,-1,-1):
				if self.polynomial_coeffs_m[j][k] != 0:
					break

			# Find the largest localising matrix that can be support by the defined moment sequence
			for i in range(self.M):
				if not self.is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=self.d))):
					break

			mm_degree = self.num_mms()-1

			# Allocate space for local matrix
			lm_dim = len(generate_unsorted_idx_list(M=(mm_degree-1)//2, d=self.d))
			self.localising_matrices_mj.append(cp.Variable((lm_dim,lm_dim), PSD=True))


		# Create empty localizing matrices for BJs
		num_constraint_polys_bj = len(self.polynomial_coeffs_b)
		for j in range(num_constraint_polys_bj):
			# Generate list of moment indexes for q alpha
			idx_list = generate_unsorted_idx_list(M=self.q_alpha, d=self.d)
			idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			# Find the largest index in the q polynomial coeff that isn't 0
			for k in range(len(idx_list)-1,-1,-1):
				if self.polynomial_coeffs_b[j][k] != 0:
					break

			# Find the largest localising matrix that can be support by the defined moment sequence
			for i in range(self.M):
				if not self.is_lm_valid(idx_list[k], len(generate_unsorted_idx_list(M=i, d=self.d))):
					break
			
			mm_degree = self.num_mms()-1

			# Allocate space for local matrix
			lm_dim = len(generate_unsorted_idx_list(M=(mm_degree-1)//2, d=self.d))
			self.localising_matrices_bj.append(cp.Variable((lm_dim,lm_dim), PSD=True))


		# self.lm_mj = []
		# self.lm_bj_inner = []
		# self.lm_bj_outer = []

		# # Create empty localizing matrices for MJs
		# num_constraint_polys_mj = len(self.polynomial_coeffs_m)
		# for j in range(num_constraint_polys_mj):
		# 	self.lm_mj.append([])
		# 	# Calculate the max number of localizing matrices
		# 	for i in range(len(self.mm_mj)):

		# 		# Generate list of moment indexes for q alpha
		# 		idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
		# 		idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
		# 		# Find the largest index in the q polynomial coeff that isn't 0
		# 		for k in range(len(idx_list)-1,-1,-1):
		# 			if self.polynomial_coeffs_m[j][k] != 0:
		# 				break

		# 		# Check to see if this lm contains valid indexes
		# 		if self.is_lm_valid(idx_list[k], self.mm_mj[i].shape[0]):
		# 			# Allocate space for local matrix
		# 			self.lm_mj[-1].append(cp.Variable((self.mm_mj[i].shape[0],self.mm_mj[i].shape[1]), PSD=True))
		# 		else:
		# 			break
		


		# # Create empty localizing matrices for BJs
		# num_constraint_polys_bj = len(self.polynomial_coeffs_bj_inner)
		# for j in range(num_constraint_polys_bj):
		# 	self.lm_bj_inner.append([])
		# 	self.lm_bj_outer.append([])
		# 	# Calculate the max number of localizing matrices
		# 	for i in range(len(self.mm_bj)):

		# 		# Generate list of moment indexes for q alpha
		# 		idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
		# 		idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
		# 		# Find the largest index in the q polynomial coeff that isn't 0
		# 		for k in range(len(idx_list)-1,-1,-1):
		# 			if self.polynomial_coeffs_bj_inner[j][k] != 0:
		# 				break

		# 		# Check to see if this lm contains valid indexes
		# 		if self.is_lm_valid(idx_list[k], self.mm_bj[i].shape[0]):
		# 			# Allocate space for local matrix
		# 			self.lm_bj_inner[-1].append(cp.Variable((self.mm_bj[i].shape[0],self.mm_bj[i].shape[1]), PSD=True))
		# 			self.lm_bj_outer[-1].append(cp.Variable((self.mm_bj[i].shape[0],self.mm_bj[i].shape[1]), PSD=True))
		# 		else:
		# 			break
	

					
	def mm_constraints(self):
		# constraints = []

		# for mm_idx in range(len(self.mm_mj)):
		# 	for i in range(self.mm_mj[mm_idx].shape[0]):
		# 		for j in range(self.mm_mj[mm_idx].shape[1]):

		# 			moment_idx = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]

		# 			constraints.append(self.mm_mj[mm_idx][i][j]
		# 				               == self.mj[self.idx_list.index(moment_idx)])
		# 			constraints.append(self.mm_bj[mm_idx][i][j]
		# 				               == self.bj[self.idx_list.index(moment_idx)])	
					
		# 			# Create dictionary entry for LMs
		# 			self.mm_dict[str(mm_idx)+'_'+str(i)+'_'+str(j)] = moment_idx

		constraints = []

		for i in range(self.moment_matrix_mj.shape[0]):
			for j in range(self.moment_matrix_mj.shape[1]):
				moment_idx = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]

				constraints.append(self.moment_matrix_mj[i][j]
						               == self.mj[self.idx_list.index(moment_idx)])
				constraints.append(self.moment_matrix_bj[i][j]
						               == self.bj[self.idx_list.index(moment_idx)])	

				# Create dictionary entry for LMs
				self.mm_dict[str(i)+'_'+str(j)] = moment_idx

		return constraints



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
					idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
					idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

					# Build constraint instruction as a string
					constriant_str_mj = ''
					constriant_str_mj += 'self.localising_matrices_mj['+str(cur_lm)+']['+str(i)+']['+str(j)+']=='

					for alpha in range(len(idx_list)):
						# Get moment index by summing alpha index with moment matrix index
						lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
						if self.polynomial_coeffs_m[cur_lm][alpha] != 0:
							# Generate constraint string 
							constriant_str_mj += 'self.polynomial_coeffs_m[cur_lm]['+str(alpha)+']*'
							constriant_str_mj += 'self.mj['+str(self.idx_list.index(lm_idx))+']+'

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
					idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
					idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

					# Build constraint instruction as a string
					constriant_str_bj = ''
					constriant_str_bj += 'self.localising_matrices_bj['+str(cur_lm)+']['+str(i)+']['+str(j)+']=='

					for alpha in range(len(idx_list)):
						# Get moment index by summing alpha index with moment matrix index
						lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]
						if self.polynomial_coeffs_b[cur_lm][alpha] != 0:
							# Generate constraint string 
							constriant_str_bj += 'self.polynomial_coeffs_b[cur_lm]['+str(alpha)+']*'
							constriant_str_bj += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'

					# Remove ending '+' 
					constriant_str_bj = constriant_str_bj[:-1]
					# Append constraint to list and execute 
					command_bj = 'constraints.append('+constriant_str_bj+')'
					exec(command_bj)



		# constraints = []

		# # Obtain the contraints for the MJ localizing matrices
		# num_constraint_polys_mj = len(self.polynomial_coeffs_m)
		# # Calculate the number of localising matrices per constraint polynomial
		# num_lms_mj = int(len(self.lm_mj)/num_constraint_polys_mj)


		# for cur_constraint_poly in range(num_constraint_polys_mj):
		# 	# Calculate the number of localising matrices for the current constraint polynomial
		# 	num_lms_mj = len(self.lm_mj[cur_constraint_poly])
		# 	for cur_mat in range(num_lms_mj):
		# 		# Link to moments constraints
		# 		for i in range(self.lm_mj[cur_constraint_poly][cur_mat].shape[0]):
		# 			for j in range(self.lm_mj[cur_constraint_poly][cur_mat].shape[1]):
						
		# 				# Get index from corresponding moment matrix
		# 				mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

		# 				# list of alpha indexes
		# 				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
		# 				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

		# 				# Build constraint instruction as a string
		# 				constriant_str_mj = ''
		# 				constriant_str_mj += 'self.lm_mj['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='

		# 				for alpha in range(len(idx_list)):

		# 					# Get moment index by summing alpha index with moment matrix index
		# 					lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

		# 					if self.polynomial_coeffs_m[cur_constraint_poly][alpha] != 0:
		# 						# Generate constraint string 
		# 						constriant_str_mj += 'self.polynomial_coeffs_m[cur_constraint_poly]['+str(alpha)+']*'
		# 						constriant_str_mj += 'self.mj['+str(self.idx_list.index(lm_idx))+']+'

		# 				# Remove ending '+' 
		# 				constriant_str_mj = constriant_str_mj[:-1]
	
		# 				# Append constraint to list and execute 
		# 				command_mj = 'constraints.append('+constriant_str_mj+')'

		# 				exec(command_mj)


		# # Obtain the contraints for the BJ (Inner) localizing matrices
		# num_constraint_polys_bj_inner = len(self.polynomial_coeffs_bj_inner)		#Inner and Outer are negatives of each other so lengths are the same
		# # Calculate the number of localising matrices per constraint polynomial

		# for cur_constraint_poly in range(num_constraint_polys_bj_inner):
		# 	num_lms_bj_inner = len(self.lm_bj_inner[cur_constraint_poly])
		# 	for cur_mat in range(num_lms_bj_inner):
		# 		# Link to moments constraints
		# 		for i in range(self.lm_bj_inner[cur_constraint_poly][cur_mat].shape[0]):
		# 			for j in range(self.lm_bj_inner[cur_constraint_poly][cur_mat].shape[1]):
						
		# 				# Get index from corresponding moment matrix
		# 				mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

		# 				# list of alpha indexes
		# 				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
		# 				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

		# 				# Build constraint instruction as a string
		# 				constriant_str_bj_inner = ''
		# 				constriant_str_bj_inner += 'self.lm_bj_inner['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='
						
		# 				for alpha in range(len(idx_list)):

		# 					# Get moment index by summing alpha index with moment matrix index
		# 					lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

		# 					if self.polynomial_coeffs_bj_inner[cur_constraint_poly][alpha] != 0:
		# 						constriant_str_bj_inner += 'self.polynomial_coeffs_bj_inner[cur_constraint_poly]['+str(alpha)+']*'
		# 						constriant_str_bj_inner += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'
							
		# 				# Remove ending '+' 
		# 				constriant_str_bj_inner = constriant_str_bj_inner[:-1]
						
		# 				# Append constraint to list and execute 
		# 				command_bj_inner = 'constraints.append('+constriant_str_bj_inner+')'

		# 				exec(command_bj_inner)
		



		# # Obtain the contraints for the BJ (Outer) localizing matrices
		# num_constraint_polys_bj_outer = len(self.polynomial_coeffs_bj_outer)		#Inner and Outer are negatives of each other so lengths are the same
		# # Calculate the number of localising matrices per constraint polynomial

		# for cur_constraint_poly in range(num_constraint_polys_bj_outer):
		# 	num_lms_bj_outer = len(self.lm_bj_outer[cur_constraint_poly])
		# 	for cur_mat in range(num_lms_bj_outer):
		# 		# Link to moments constraints
		# 		for i in range(self.lm_bj_outer[cur_constraint_poly][cur_mat].shape[0]):
		# 			for j in range(self.lm_bj_outer[cur_constraint_poly][cur_mat].shape[1]):
						
		# 				# Get index from corresponding moment matrix
		# 				mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

		# 				# list of alpha indexes
		# 				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
		# 				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

		# 				# Build constraint instruction as a string
		# 				constriant_str_bj_outer = ''
		# 				constriant_str_bj_outer += 'self.lm_bj_outer['+str(cur_constraint_poly)+']['+str(cur_mat)+']['+str(i)+']['+str(j)+']=='

		# 				for alpha in range(len(idx_list)):

		# 					# Get moment index by summing alpha index with moment matrix index
		# 					lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

		# 					if self.polynomial_coeffs_bj_outer[cur_constraint_poly][alpha] != 0:
		# 						constriant_str_bj_outer += 'self.polynomial_coeffs_bj_outer[cur_constraint_poly]['+str(alpha)+']*'
		# 						constriant_str_bj_outer += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'

		# 				# Remove ending '+' 
		# 				constriant_str_bj_outer = constriant_str_bj_outer[:-1]
						
		# 				# Append constraint to list and execute 
		# 				command_bj_outer = 'constraints.append('+constriant_str_bj_outer+')'

		# 				exec(command_bj_outer)

		return constraints




	# Martingale constraints
	# states: [x, y, gamma (g), t, cos(g), sin(g)]
	# *Refer to RN notebook 12/4 for generator  
	def martingale_constraints(self):

		constraints = []

		################ Degree 0 #########################################################
		constraints.append(-self.bj[0] + 1 == 0)
		# constraints.append(self.mj[0] >= 0)

		################ Degree 1 #########################################################
		start = next(i for i, v in enumerate(self.idx_list)
						if np.sum(self.idx_list[i]) == 1)
		end = next(i for i, v in enumerate(self.idx_list)
					if np.sum(self.idx_list[i]) == 2)

		for n in range(start, end):
			invalid_idx = False
			f = copy.deepcopy(self.idx_list[n])  # Get the test function

			# For each test function perform the following derivatives:
			# dt
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
			# dt, dy^2
			dt = mono_derivative(f, [1])
			dy_2 = mono_derivative(f, [0, 0])

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
			if dy_2[0] != 0:
				# generator term: 0.5  * d^2f/dy^2
				if not dy_2[1] in self.idx_list:
					invalid_idx = True
				constriant_str += '0.5*dy_2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dy_2[1])]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			if not invalid_idx:
				exec(command)

		print('Num martingale constraints:', len(constraints))
		return constraints




if __name__ == '__main__':


	state_dim = 2 		# [y, t]
	q_polynomial_deg = 2
	T = 5.
	idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

	y_lower = 0.
	y_upper = 1.
	t_lower = 0.
	t_upper = T
	# Polynomials for safe set:
	# -y^2 + (l+u)y - lu
    # -y^2 + y >= 0
	pcs_y = np.zeros(len(idx_list)).tolist()
	# pcs_y[idx_list.index([2, 0])] = -1
	# pcs_y[idx_list.index([1, 0])] = 1
	pcs_y[idx_list.index([2, 0])] = -1
	pcs_y[idx_list.index([1, 0])] = y_lower + y_upper
	pcs_y[idx_list.index([0, 0])] = -y_lower * y_upper

	# -t^2 + (l+u)t - lu
    # -t^2 + Tt >= 0
	pcs_t = np.zeros(len(idx_list)).tolist()
	# pcs_t[idx_list.index([0, 2])] = -1
	# pcs_t[idx_list.index([0, 1])] = T
	pcs_t[idx_list.index([0, 2])] = -1
	pcs_t[idx_list.index([0, 1])] = t_lower + t_upper
	pcs_t[idx_list.index([0, 0])] = -t_lower * t_upper

	pcs_m = [pcs_y, pcs_t]
	pcs_b = [pcs_t, pcs_y, [-i for i in pcs_y]]

	num_moments = len(generate_unsorted_idx_list(M=8,d=2))
	print('Number of moments:', num_moments)

	# System parameters:
	ks = 5.
	ms = 1.
	kc = 1.
	starting_state = [0.5, 0.]

	values = []

	for order in range(5,6):

		ets = escape_time_solver(d=state_dim, M=num_moments, q_alpha=q_polynomial_deg, 
								pcs_m=pcs_m, pcs_b=pcs_b, y0=starting_state,
								max_opt_iters=500000, acc_lookback=10, sys_params=[ks,ms,kc],
								solver_verbose=True, exit_time_order=order+1, solver='SCS', 
								warm_start=False)

		start_time = time.time()
		values.append(ets.solve(mode='Maximize'))
		print('end time:',time.time()-start_time)

	#pickle.dump(ets, open('results/SpringMassDamper_M6_itr2M.pkl', 'wb'))
	