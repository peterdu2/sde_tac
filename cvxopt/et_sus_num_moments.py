import cvxpy as cp 
import numpy as np 
import time
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative
import copy
import math
import pickle

# Constants for indexing state variables
X1_IDX = 0
X2_IDX = 1


def is_psd(x):
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:

	def __init__(self, d, M, pcs_m, pcs_b, sys_params, q_alpha=5, y0=0.2, lm_vis=True, max_opt_iters=5000,
			     acc_lookback=5, warm_start=True, solver_verbose=True):
		self.d = d # Dimension of system 
		self.M = M # Number of moments (multi-indexes)

		# Generate list of moment indexes
		idx_list = generate_unsorted_idx_list(M=100,d=d)
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
		self.lm_bj2 = None

		# Moment idx dictionary (Used for creating LMs)
		self.mm_dict = {}	# Key: Str(moment matrix idx,_,i,_,j)
							# Eg:  Moment matrix 2, entry 1,3 -> 2_1_3
							# Value: moment index (multi-index)

		# Highest degree of monomials for polynomial q
		self.q_alpha = q_alpha

		# Polynomials used to define localizing matrices
		self.polynomial_coeffs_m = pcs_m
		self.polynomial_coeffs_b = pcs_b

		self.y0 = y0

		# Visualize localizing matrices
		self.lm_vis = lm_vis
		self.max_opt_iters = max_opt_iters
		self.acc_lookback = acc_lookback
		self.warm_start = warm_start
		self.solver_verbose = solver_verbose

		# System parameters
		self.ks = sys_params[0]
		self.ms = sys_params[1]
		self.kc = sys_params[2]
		self.g = 9.81



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
		print('Number of localising matrices:', len(self.lm_mj))
		print('Time to create localising matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.lm_constraints()
		print('Time to create lm constraints:', time.time()-start_time)

		start_time = time.time()
		constraints += self.martingale_constraints()
		print('Time to create martingale constraints:', time.time()-start_time)

		start_time = time.time()
		obj = cp.Maximize(self.mj[0]) if mode == 'Maximize' else cp.Minimize(self.mj[0])
		print('Objective', obj)
		print('Time to create objective:', time.time()-start_time)

		start_time = time.time()
		prob = cp.Problem(obj, constraints)
		print('Time to create problem object:', time.time()-start_time)

		# prob.solve(max_iters=10000, verbose=True)
		#prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.SCS, 
			    #    acceleration_lookback=self.acc_lookback)
		prob.solve(verbose=self.solver_verbose, solver=cp.MOSEK)

		# # print('MJs')
		# # for mj in self.mj:
		# # 	print(mj.value)
		# # print('\nBJs')
		# # for bj in self.bj:
		# # 	print(bj.value)

		# print(' ')

		print(prob.value)
		print(prob.status)




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
		self.lm_bj = []
		self.lm_bj2 = []

		num_constraint_polys = len(self.polynomial_coeffs_m)

		for j in range(num_constraint_polys):
			# Calculate the max number of localizing matrices
			# for i in range(len(self.mm_mj)):
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
					self.lm_mj.append(cp.Variable((self.mm_mj[i].shape[0],self.mm_mj[i].shape[1]), PSD=True))
					self.lm_bj.append(cp.Variable((self.mm_mj[i].shape[0],self.mm_mj[i].shape[1]), PSD=True))
					self.lm_bj2.append(cp.Variable((self.mm_mj[i].shape[0],self.mm_mj[i].shape[1]), PSD=True))
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

		num_constraint_polys = len(self.polynomial_coeffs_m)
		# Calculate the number of localising matrices per constraint polynomial
		num_lms = int(len(self.lm_mj)/num_constraint_polys)

		for cur_constraint_poly in range(num_constraint_polys):
			for cur_mat in range(num_lms):

				cur_local_matrix = copy.copy(cur_mat) + cur_constraint_poly*num_lms

				##################################################################################
				# Visualize localizing matrix
				if self.lm_vis:
					vis_lm_matrix_mj = [['' for j in range(self.lm_mj[cur_local_matrix].shape[0])] for i in range(self.lm_mj[cur_local_matrix].shape[0])]
					vis_lm_matrix_bj = [['' for j in range(self.lm_bj[cur_local_matrix].shape[0])] for i in range(self.lm_bj[cur_local_matrix].shape[0])]
				##################################################################################
				

				# Link to moments constraints
				for i in range(self.lm_mj[cur_local_matrix].shape[0]):
					for j in range(self.lm_mj[cur_local_matrix].shape[1]):
						
						# Get index from corresponding moment matrix
						mm_idx = self.mm_dict[str(cur_mat)+'_'+str(i)+'_'+str(j)]

						# list of alpha indexes
						idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
						idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

						# Build constraint instruction as a string
						constriant_str_mj = ''
						constriant_str_mj += 'self.lm_mj['+str(cur_local_matrix)+']['+str(i)+']['+str(j)+']=='
						constriant_str_bj = ''
						constriant_str_bj += 'self.lm_bj['+str(cur_local_matrix)+']['+str(i)+']['+str(j)+']=='
						constriant_str_bj2 = ''
						constriant_str_bj2 += 'self.lm_bj2['+str(cur_local_matrix)+']['+str(i)+']['+str(j)+']=='

						for alpha in range(len(idx_list)):

							# Get moment index by summing alpha index with moment matrix index
							lm_idx = [sum(x) for x in zip(idx_list[alpha], mm_idx)]

							if self.polynomial_coeffs_m[cur_constraint_poly][alpha] != 0:
								# Generate constraint string 
								constriant_str_mj += 'self.polynomial_coeffs_m[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_mj += 'self.mj['+str(self.idx_list.index(lm_idx))+']+'

							if self.polynomial_coeffs_b[cur_constraint_poly][alpha] != 0:
								constriant_str_bj += 'self.polynomial_coeffs_b[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_bj += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'
								constriant_str_bj2 += 'self.polynomial_coeffs_m[cur_constraint_poly]['+str(alpha)+']*'
								constriant_str_bj2 += 'self.bj['+str(self.idx_list.index(lm_idx))+']+'


							##################################################################################
							# localizing matrix visualization
							if self.lm_vis:
								if self.polynomial_coeffs_m[cur_constraint_poly][alpha] != 0:
									vis_lm_matrix_mj[i][j] += str(self.polynomial_coeffs_m[cur_constraint_poly][alpha])
									vis_lm_matrix_bj[i][j] += str(self.polynomial_coeffs_b[cur_constraint_poly][alpha])
									vis_lm_matrix_mj[i][j] += '*'
									vis_lm_matrix_bj[i][j] += '*'
									for digit in lm_idx:
										vis_lm_matrix_mj[i][j] += str(digit)
										vis_lm_matrix_bj[i][j] += str(digit)
									if alpha < len(idx_list)-1:
										vis_lm_matrix_mj[i][j] += '+'
										vis_lm_matrix_bj[i][j] += '+'
							##################################################################################

						# Remove ending '+' 
						constriant_str_mj = constriant_str_mj[:-1]
						constriant_str_bj = constriant_str_bj[:-1]
						constriant_str_bj2 = constriant_str_bj2[:-1]
						
						# Append constraint to list and execute 
						command_mj = 'constraints.append('+constriant_str_mj+')'
						command_bj = 'constraints.append('+constriant_str_bj+')'
						command_bj2 = 'constraints.append('+constriant_str_bj2+')'

						exec(command_mj)
						exec(command_bj)
						exec(command_bj2)

				# Visualize local matrix
				if self.lm_vis:
					print('LM MJ',cur_mat, vis_lm_matrix_mj) 
					print('LM BJ',cur_mat, vis_lm_matrix_bj) 


		return constraints




	# Martingale constraints for a 6 dimensional turtlebot system
	# states: [x, y, gamma (g), t, cos(g), sin(g)]
	# *Refer to RN notebook 12/4 for generator  
	def martingale_constraints(self):

		constraints = []

		################ Degree 0 #########################################################
		constraints.append(-self.bj[0] + 1 == 0)


		################ Degree 1 #########################################################
		start = next(i for i,v in enumerate(self.idx_list) if np.sum(self.idx_list[i])==1)
		end = next(i for i,v in enumerate(self.idx_list) if np.sum(self.idx_list[i])==2)

		for n in range(start,end):
			invalid_idx = False
			f = copy.deepcopy(self.idx_list[n])	# Get the test function

			# For each test function perform the following derivatives:
			# dx1, dx2, dx2^2
			dx1 = mono_derivative(f,[0])
			dx2 = mono_derivative(f,[1])
			dx2_2 = mono_derivative(f,[1,1])

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
			if dx1[0] != 0:
				# generator term: x2 * df/dx1
				dx1[1][X2_IDX] += 1
				if not dx1[1] in self.idx_list:
					invalid_idx = True
				constriant_str += 'dx1[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx1[1])]+'
			if dx2[0] != 0:
				# generator term: (-ks*x1/ms + g - kc*x1*x2/ms) * df/dx2
				t1 = copy.copy(dx2[1])
				t2 = copy.copy(dx2[1])
				t3 = copy.copy(dx2[1])
				t1[X1_IDX] += 1
				t3[X1_IDX] += 1
				t3[X2_IDX] += 1
				if not t1 in self.idx_list or not t3 in self.idx_list:
					invalid_idx = True
				constriant_str += '-self.ks*dx2[0]/self.ms*'
				constriant_str += 'self.mj[self.idx_list.index(t1)]+'
				constriant_str += 'self.g*dx2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(t2)]+'
				constriant_str += '-self.kc*dx2[0]/self.ms*'
				constriant_str += 'self.mj[self.idx_list.index(t3)]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			if not invalid_idx:
				exec(command)


		################ Degree 2 and above ###############################################
		# Get position in index list where monomial degree is 2
		start = next(i for i,v in enumerate(self.idx_list) if np.sum(self.idx_list[i])==2)
		
		# n represents the index of the monomial test function
		for n in range(start, len(self.mj)):
			invalid_idx = False
			f = copy.deepcopy(self.idx_list[n])	# Get the test function

			# For each test function perform the following derivatives:
			# dx1, dx2, dx2^2
			dx1 = mono_derivative(f,[0])
			dx2 = mono_derivative(f,[1])
			dx2_2 = mono_derivative(f,[1,1])

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
			if dx1[0] != 0:
				# generator term: x2 * df/dx1
				dx1[1][X2_IDX] += 1
				if not dx1[1] in self.idx_list:
					invalid_idx = True
				constriant_str += 'dx1[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx1[1])]+'
			if dx2[0] != 0:
				# generator term: (-ks*x1/ms + g - kc*x1*x2/ms) * df/dx2
				t1 = copy.copy(dx2[1])
				t2 = copy.copy(dx2[1])
				t3 = copy.copy(dx2[1])
				t1[X1_IDX] += 1
				t3[X1_IDX] += 1
				t3[X2_IDX] += 1
				if not t1 in self.idx_list or not t3 in self.idx_list:
					invalid_idx = True
				constriant_str += '-self.ks*dx2[0]/self.ms*'
				constriant_str += 'self.mj[self.idx_list.index(t1)]+'
				constriant_str += 'self.g*dx2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(t2)]+'
				constriant_str += '-self.kc*dx2[0]/self.ms*'
				constriant_str += 'self.mj[self.idx_list.index(t3)]+'
			if dx2_2[0] != 0:
				# generator term: 0.5 * (kc/ms)^2 * x1^2 * d^2f/dx2^2
				dx2_2[1][X1_IDX] += 2
				if not dx2_2[1] in self.idx_list:
					invalid_idx = True
				constriant_str += '0.5*dx2_2[0]*(self.kc/self.ms)**2*'
				constriant_str += 'self.mj[self.idx_list.index(dx2_2[1])]+'
			

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			if not invalid_idx:
				exec(command)

		print('Num martingale constraints:', len(constraints))
		return constraints




if __name__ == '__main__':

	# idx_list = generate_unsorted_idx_list(M=30,d=2)
	# idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	# for i in idx_list:
	# 	print(i)
	# print(len(idx_list), '\n')


	# Notes: 
	# 1) q_alpha is the highest degree of polynomials q that define the safe set
	#    the size of pcs_m and pcs_b need to match the number of monomial terms for q_alpha
	# 
	# 2) State space dimensionality: 2
	#    state = [position, velocity]

	#118 7-7 
	#120 8-7
	#151 8-8
	#188 9-9
	#229 10-10
	#230 10-10
	#231 11-10
	#274 11-11
	#323 12-12
	#376 13-13
	#494 15-15
	state_dim = 2
	num_moments = 20

	# Generate polynomial inequalities that define safe/unsafe sets
	# Ellipse: 
	#		-3.180005x1 + 3.85x1 - x1^2 - 1/9x2^2 >= 0
	# q_polynomial_deg = 2
	# num_inequalities = 1
	# idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	# idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	# pcs_m = np.zeros(len(idx_list)).tolist()
	# pcs_m[idx_list.index([0,0])] = -3.180005
	# pcs_m[idx_list.index([1,0])] = 3.85
	# pcs_m[idx_list.index([2,0])] = -1
	# pcs_m[idx_list.index([0,2])] = -1/9
	# pcs_b = [[-i if i != 0 else 0 for i in pcs_m]]
	# pcs_m = [pcs_m]
	
	# Parabola 1:
	#		-3.18 + 3.85x1 - x1^2 >= 0
	# q_polynomial_deg = 2
	# num_inequalities = 1
	# idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	# idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	# pcs_m = np.zeros(len(idx_list)).tolist()
	# pcs_m[idx_list.index([0,0])] = -3.18
	# pcs_m[idx_list.index([1,0])] = 3.85
	# pcs_m[idx_list.index([2,0])] = -1
	# pcs_b = [[-i if i != 0 else 0 for i in pcs_m]]
	# pcs_m = [pcs_m]


	# Parabola 2:
	#		-2.35 + 3.35x1 -x^2 >= 0
	q_polynomial_deg = 2
	num_inequalities = 1
	idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	pcs_m = np.zeros(len(idx_list)).tolist()
	pcs_m[idx_list.index([0,0])] = -2.35
	pcs_m[idx_list.index([1,0])] = 3.35
	pcs_m[idx_list.index([2,0])] = -1
	pcs_b = [[-i if i != 0 else 0 for i in pcs_m]]
	pcs_m = [pcs_m]


	print(pcs_m)
	print(pcs_b)


	# System parameters:
	ks = 5
	ms = 1
	kc = 0.5
	# state = [position, velocity]
	starting_state = [9.81/ks, 0.]

	ets = escape_time_solver(d=state_dim, M=num_moments, q_alpha=q_polynomial_deg, 
		                     pcs_m=pcs_m, pcs_b=pcs_b, lm_vis=False, y0=starting_state,
		                     max_opt_iters=3000000, acc_lookback=10, sys_params=[ks,ms,kc],
		                     solver_verbose=True)

	start_time = time.time()
	ets.solve(mode='Minimize')
	print('end time:',time.time()-start_time)

	#pickle.dump(ets, open('results/SMD/with_x1/M494_3000000_min.pkl', 'wb'))
	