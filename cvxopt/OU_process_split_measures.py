import cvxpy as cp 
import numpy as np 
import time
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative
import copy
import math
import pickle

# Constants for indexing state variables
POS_IDX = 0
TIME_IDX = 1


def is_psd(x):
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:

	def __init__(self, d, M, pcs_m, pcs_bt, pcs_bb, pcs_br,
				 sys_params, T=10., q_alpha=5, y0=0.2, lm_vis=True, max_opt_iters=5000,
			     acc_lookback=5, solver_verbose=True, exit_time_order=1, 
				 solver='SCS', print_moments=False):
		self.d = d # Dimension of system 
		self.M = M # Number of moments (multi-indexes)
		self.exit_time_order = exit_time_order
		self.solver = solver
		self.print_moments = print_moments

		# Generate list of moment indexes
		idx_list = generate_unsorted_idx_list(M=M,d=d)
		self.idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

		# Moment arrays
		self.mj = [cp.Variable() for i in range(len(self.idx_list))]
		# Moment arrays for the three exit measures (b_t, b_b, b_r)
		num_exit_measure_moments = len(generate_unsorted_idx_list(M=M,d=1))
		self.bt = [cp.Variable() for i in range(num_exit_measure_moments)]
		self.bb = [cp.Variable() for i in range(num_exit_measure_moments)]
		self.br = [cp.Variable() for i in range(num_exit_measure_moments)]

		# Moment matrices
		self.mm_mj = None
		self.mm_bt = None
		self.mm_bb = None
		self.mm_br = None

		# Localizing matrices
		self.lm_mj = None
		self.lm_bt = None
		self.lm_bb = None
		self.lm_br = None


		# Moment idx dictionary (Used for creating LMs)
		self.mm_dict_mj = {}	# Key: Str(moment matrix idx,_,i,_,j)
							# Eg:  Moment matrix 2, entry 1,3 -> 2_1_3
							# Value: moment index (multi-index)


		# Highest degree of monomials for polynomial q
		self.q_alpha = q_alpha

		# Polynomials used to define localizing matrices
		self.polynomial_coeffs_m = pcs_m
		self.polynomial_coeffs_bt = pcs_bt
		self.polynomial_coeffs_bb = pcs_bb
		self.polynomial_coeffs_br = pcs_br

		self.y0 = y0
		self.T = T

		# Solver parameters
		self.solver_verbose = solver_verbose
		self.max_opt_iters = max_opt_iters
		self.acc_lookback = acc_lookback

		# System parameters
		self.alpha = sys_params[0]
		self.sigma = sys_params[1]

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
		print('Time to create localising matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.lm_constraints()
		print('Time to create lm constraints:', time.time()-start_time)

		start_time = time.time()
		constraints += self.martingale_constraints()
		print('Time to create martingale constraints:', time.time()-start_time)


		#self.idx_list.index(dt[1])
		start_time = time.time()
		obj = cp.Maximize(self.exit_time_order*self.T**(self.exit_time_order-1)*self.mj[self.idx_list.index([0,self.exit_time_order-1])]) if mode == 'Maximize' else cp.Minimize(self.exit_time_order*self.T**(self.exit_time_order-1)*self.mj[self.idx_list.index([0,self.exit_time_order-1])])
		print('Objective', obj)
		print('Time to create objective:', time.time()-start_time)

		start_time = time.time()
		self.prob = cp.Problem(obj, constraints)
		print('Time to create problem object:', time.time()-start_time)

		
		if self.solver == 'SCS':
			self.prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.SCS, 
			       			acceleration_lookback=self.acc_lookback)
		else:
			self.prob.solve(verbose=self.solver_verbose,solver=cp.MOSEK)

		if self.print_moments:
			print('Occupation Measure Moments')
			for mj in self.mj[:5]:
				print(mj.value)
			print('\nExit Measure Moments')
			print('bt:', self.bt[0].value)
			print('bb:', self.bb[0].value)
			print('br:', self.br[0].value)

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
		# Create moment matrix for occupation measure
		self.mm_mj = []
		max_num_mms = self.num_mms()
		# Instantiate the moment matrices 
		for i in range(max_num_mms):

			# Generate list of moment indexes for current moment matrix
			idx_list = generate_unsorted_idx_list(M=i,d=self.d)
			idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			
			# Allocate space for moment matrix
			self.mm_mj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))

		
		# Create moment matrices for exit measure (b_t, b_b, b_r)
		num_mms = self.M//2+1
		self.mm_bt = [cp.Variable((i,i), PSD=True) for i in range(1,num_mms+1)]
		self.mm_bb = [cp.Variable((i,i), PSD=True) for i in range(1,num_mms+1)]
		self.mm_br = [cp.Variable((i,i), PSD=True) for i in range(1,num_mms+1)]



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

		# Create empty localizing matrices for MJs (occupation measure)
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
		
		# Create empty localizing matrices for exit measures (b_t, b_b, b_r)
		num_mms = self.M//2+1

		# b_t
		for num_lm in range(num_mms,0,-1):
			max_mm_idx = 2*(self.mm_bt[num_lm-1].shape[0]-1) # largest index in moment matrix
			max_alpha = len(self.polynomial_coeffs_bt)-1	# largest alpha (index of polynomial coeffs)
			max_lm_idx = max_mm_idx + max_alpha		# largest index in localising matrix
			if max_lm_idx <= len(self.bt)-1:		# check if largest lm index is within moment arrays 
				break
		self.lm_bt = [cp.Variable((i,i), PSD=True) for i in range(1,num_lm+1)]

		# b_b
		for num_lm in range(num_mms,0,-1):
			max_mm_idx = 2*(self.mm_bb[num_lm-1].shape[0]-1) # largest index in moment matrix
			max_alpha = len(self.polynomial_coeffs_bb)-1	# largest alpha (index of polynomial coeffs)
			max_lm_idx = max_mm_idx + max_alpha		# largest index in localising matrix
			if max_lm_idx <= len(self.bb)-1:		# check if largest lm index is within moment arrays 
				break
		self.lm_bb = [cp.Variable((i,i), PSD=True) for i in range(1,num_lm+1)]

		# b_r
		for num_lm in range(num_mms,0,-1):
			max_mm_idx = 2*(self.mm_br[num_lm-1].shape[0]-1) # largest index in moment matrix
			max_alpha = len(self.polynomial_coeffs_br)-1	# largest alpha (index of polynomial coeffs)
			max_lm_idx = max_mm_idx + max_alpha		# largest index in localising matrix
			if max_lm_idx <= len(self.br)-1:		# check if largest lm index is within moment arrays 
				break
		self.lm_br = [cp.Variable((i,i), PSD=True) for i in range(1,num_lm+1)]

		
					
	def mm_constraints(self):
		constraints = []

		# MM constraints for occupation measure moment matrix
		for mm_idx in range(len(self.mm_mj)):
			for i in range(self.mm_mj[mm_idx].shape[0]):
				for j in range(self.mm_mj[mm_idx].shape[1]):
					moment_idx = [sum(x) for x in zip(self.idx_list[i], self.idx_list[j])]
					constraints.append(self.mm_mj[mm_idx][i][j]
						               == self.mj[self.idx_list.index(moment_idx)])
					# Create dictionary entry for LMs
					self.mm_dict_mj[str(mm_idx)+'_'+str(i)+'_'+str(j)] = moment_idx
		
		# MM constraints for exit measures (b_t, b_b, b_r)
		num_mms = self.M//2+1
		for cur_mat in range(num_mms):
			# Link to moments constraints
			for i in range(cur_mat+1):
				for j in range(cur_mat+1):
					moment_idx = i+j
					constraints.append(self.mm_bt[cur_mat][i][j] == self.bt[moment_idx])
					constraints.append(self.mm_bb[cur_mat][i][j] == self.bb[moment_idx])
					constraints.append(self.mm_br[cur_mat][i][j] == self.br[moment_idx])

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
						mm_idx = self.mm_dict_mj[str(cur_mat)+'_'+str(i)+'_'+str(j)]

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


		
		# localizing matrix constraints for exit measure LMs
		for cur_mat in range(len(self.lm_bt)):
			# Link to moments constraints
			for i in range(cur_mat+1):
				for j in range(cur_mat+1):
					mm_idx = i+j

					constraints.append(self.lm_bt[cur_mat][i][j] 
									   == self.polynomial_coeffs_bt[0]*self.bt[mm_idx]	#alpha=0
									   + self.polynomial_coeffs_bt[1]*self.bt[mm_idx+1]	#alpha=1
									   + self.polynomial_coeffs_bt[2]*self.bt[mm_idx+2])	#alpha=2

					constraints.append(self.lm_bb[cur_mat][i][j] 
									   == self.polynomial_coeffs_bb[0]*self.bb[mm_idx]	#alpha=0
									   + self.polynomial_coeffs_bb[1]*self.bb[mm_idx+1]	#alpha=1
									   + self.polynomial_coeffs_bb[2]*self.bb[mm_idx+2])	#alpha=2
					
					constraints.append(self.lm_br[cur_mat][i][j] 
									   == self.polynomial_coeffs_br[0]*self.br[mm_idx]	#alpha=0
									   + self.polynomial_coeffs_br[1]*self.br[mm_idx+1]	#alpha=1
									   + self.polynomial_coeffs_br[2]*self.br[mm_idx+2])	#alpha=2



		return constraints




	# Martingale constraints for a 2 dimensional OU systems
	# States: [position, time]
	def martingale_constraints(self):

		constraints = []

		################ Degree 0 #########################################################
		constraints.append(100*(self.bt[0] + self.bb[0] + self.br[0]) == 100)
		constraints.append(self.bt[0] >= 0)
		constraints.append(self.bb[0] >= 0)
		constraints.append(self.br[0] >= 0)
		constraints.append(self.bt[0] <= 1)
		constraints.append(self.bb[0] <= 1)
		constraints.append(self.br[0] <= 1)
		constraints.append(self.mj[0] >= 0)
		constraints.append(self.mj[self.idx_list.index([2,0])] >= 0)


		################ Higher degrees ###################################################
		for f in self.idx_list[1:]:
			
			m = f[1]
			n = f[0]
		
			idx_list_1d = generate_unsorted_idx_list(M=self.M,d=1)
			idx_list_1d = sorted(idx_list_1d, key=cmp_to_key(index_comparison))

			if m >= 1 and n >= 2:
				constraints.append( 1.*((m/self.T)*self.mj[self.idx_list.index([n,m-1])] 
									- self.alpha*n*self.mj[self.idx_list.index([n,m])]
									+ (0.5*n*(n-1)*self.sigma**2)*self.mj[self.idx_list.index([n-2,m])]
									- self.bb[idx_list_1d.index([m])]
									- self.br[idx_list_1d.index([n])]) == 0)

			elif m >= 1 and n == 1:
				constraints.append( 1.*((m/self.T)*self.mj[self.idx_list.index([n,m-1])] 
							        - self.alpha*n*self.mj[self.idx_list.index([n,m])]
									- self.bb[idx_list_1d.index([m])]
									- self.br[idx_list_1d.index([n])]) == 0)
				
			elif m >= 1 and n == 0:
				constraints.append( 1.*((m/self.T)*self.mj[self.idx_list.index([n,m-1])] 
								    - self.bt[idx_list_1d.index([m])]
									- self.bb[idx_list_1d.index([m])]
									- self.br[idx_list_1d.index([n])]) == 0)
			elif m == 0 and n >= 2:
				constraints.append( 1.*((0.5*n*(n-1)*self.sigma**2)*self.mj[self.idx_list.index([n-2,m])]
									- self.alpha*n*self.mj[self.idx_list.index([n,m])]
								    + self.y0**n
									- self.bb[idx_list_1d.index([m])]
									- self.br[idx_list_1d.index([n])]) == 0)
			elif m == 0 and n == 1:
				constraints.append( 1.*(- self.alpha*n*self.mj[self.idx_list.index([n,m])]
								    + self.y0**n
									- self.bb[idx_list_1d.index([m])]
									- self.br[idx_list_1d.index([n])]) == 0)


		print('Num martingale constraints:', len(constraints))
		return constraints




if __name__ == '__main__':

	state_dim = 2
	q_polynomial_deg = 2
	idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))


	# Safe set: 
	# 2.35 >= x1 >= 1 ---->    x1 - 1 >= 0 and -x1 + 2.35 >=0
	# -(t-t_end)(t-t_start) >= 0

	# x >= 1
	pcs_x_lower = np.zeros(len(idx_list)).tolist()
	pcs_x_lower[idx_list.index([0,0])] = -1.
	pcs_x_lower[idx_list.index([1,0])] = 1.

	# x <= 10
	pcs_x_upper = np.zeros(len(idx_list)).tolist()
	pcs_x_upper[idx_list.index([0,0])] = 10.
	pcs_x_upper[idx_list.index([1,0])] = -1.

	# -10 + 11x - x^2 >= 0	lower bound x = 1
	#  10x + -x^2 >= 0 lower bound x = 0
	x_end=1.
	x_start=10.
	pcs_x_quad = np.zeros(len(idx_list)).tolist()
	pcs_x_quad[idx_list.index([0,0])] = -x_end*x_start
	pcs_x_quad[idx_list.index([1,0])] = (x_end+x_start)
	pcs_x_quad[idx_list.index([2,0])] = -1

	# t >= 0
	pcs_t_lower = np.zeros(len(idx_list)).tolist()
	pcs_t_lower[idx_list.index([0,0])] = 0.
	pcs_t_lower[idx_list.index([0,1])] = 1.

	# t <= 10
	pcs_t_upper = np.zeros(len(idx_list)).tolist()
	pcs_t_upper[idx_list.index([0,0])] = 10.
	pcs_t_upper[idx_list.index([0,1])] = -1.

	t_end=30.
	t_start=0.
	pcs_t_quad = np.zeros(len(idx_list)).tolist()
	pcs_t_quad[idx_list.index([0,0])] = -t_end*t_start
	pcs_t_quad[idx_list.index([0,1])] = (t_end+t_start)
	pcs_t_quad[idx_list.index([0,2])] = -1


	# Occupation measure constraint polynomials
	#pcs_m = [pcs_x_lower, pcs_x_upper, pcs_t_lower, pcs_t_upper]
	pcs_m = [pcs_x_quad, pcs_t_quad]
	#pcs_m = [pcs_x_quad, pcs_t_lower, pcs_t_upper]
	#pcs_m = [pcs_x_lower, pcs_x_upper, pcs_t_quad]

	
	# exit measure polynomials
	idx_list = generate_unsorted_idx_list(M=2,d=1)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	bt_end=30.
	bt_start=0.
	pcs_bt_quad = np.zeros(len(idx_list)).tolist()
	pcs_bt_quad[idx_list.index([0])] = -bt_end*bt_start
	pcs_bt_quad[idx_list.index([1])] = (bt_end+bt_start)
	pcs_bt_quad[idx_list.index([2])] = -1

	bb_end=30.
	bb_start=0.
	pcs_bb_quad = np.zeros(len(idx_list)).tolist()
	pcs_bb_quad[idx_list.index([0])] = -bb_end*bb_start
	pcs_bb_quad[idx_list.index([1])] = (bb_end+bb_start)
	pcs_bb_quad[idx_list.index([2])] = -1

	br_end=1.
	br_start=10.
	pcs_br_quad = np.zeros(len(idx_list)).tolist()
	pcs_br_quad[idx_list.index([0])] = -br_end*br_start
	pcs_br_quad[idx_list.index([1])] = (br_end+br_start)
	pcs_br_quad[idx_list.index([2])] = -1



	moment_degree = 8

	# System parameters:
	alpha = 0.525
	sigma = 0.688
	# alpha = 1.
	# sigma = 1.3


	ets = escape_time_solver(d=state_dim, M=moment_degree, q_alpha=q_polynomial_deg, 
		                     pcs_m=pcs_m, pcs_bt=pcs_bt_quad, pcs_bb=pcs_bb_quad, 
							 pcs_br=pcs_br_quad, print_moments=True, y0=9.06,
		                     max_opt_iters=1000000, acc_lookback=10, sys_params=[alpha, sigma],
		                     solver_verbose=True, exit_time_order=1, solver='SCS')

	start_time = time.time()
	ets.solve(mode='Maximize')
	print('end time:',time.time()-start_time)

	#pickle.dump(ets, open('results/SMD/time_space_SS/M265_2000000_acc5_min_deg1.pkl', 'wb'))
	