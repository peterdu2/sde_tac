import cvxpy as cp 
import numpy as np 
import time
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative
import copy
import math
import pickle
import superscs

# Constants for indexing state variables
X_IDX = 0
Y_IDX = 1
G_IDX = 2
COSG_IDX = 3
SING_IDX = 4


def is_psd(x):
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:

	def __init__(self, d, M, pcs_m, pcs_b, q_alpha=5, y0=0.2, lm_vis=True, max_opt_iters=5000, solver_verbose=True):
		self.d = d # Dimension of system 
		self.M = M # Number of moments (multi-indexes)

		# Generate list of moment indexes
		idx_list = generate_unsorted_idx_list(M=M,d=d)
		self.idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

		# Moment arrays
		# self.mj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		# self.bj = cp.Variable(len(self.idx_list))#[cp.Variable() for i in range(len(self.idx_list))]
		self.mj = [cp.Variable() for i in range(len(self.idx_list))]
		self.bj = [cp.Variable() for i in range(len(self.idx_list))]

		self.mj[0].value = 10.0
		self.bj[0].value = 1.0
		# for i in range(len(self.mj)):
		# 	self.mj[i].value=i
		# 	self.bj[i].value=i

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

		# Misc Params
		self.lm_vis = lm_vis
		self.max_opt_iters = max_opt_iters
		self.solver_verbose = solver_verbose



	def solve(self, mode='Maximize'):
		constraints = []

		start_time = time.time()
		self.create_moment_matrix()
		print('Number of moment matrices:', len(self.mm_mj))
		print('Time to create moment matrix:', time.time()-start_time)

		start_time = time.time()
		constraints += self.mm_constraints()
		self.mm_constraints()
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

		#print(self.mj[0].value)
		#self.mj[0].value=7.0
		#print(self.mj[0].value)
		# prob.solve(max_iters=10000, verbose=True)
		#prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, solver=cp.MOSEK)

		prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, 
				   solver=cp.SCS, warm_start=True, acceleration_lookback=5)
		# prob.solve(max_iters=self.max_opt_iters, verbose=self.solver_verbose, 
		# 		   solver=cp.SUPER_SCS)

		# # print('MJs')
		# # for mj in self.mj:
		# # 	print(mj.value)
		# # print('\nBJs')
		# # for bj in self.bj:
		# # 	print(bj.value)

		# print(' ')

		print(prob.value)
		print(prob.status)



	def create_moment_matrix(self):
		self.mm_mj = []
		self.mm_bj = []

		# Calculate the max number of moment matrices
		for i in range(self.M+1):

			# Generate list of moment indexes for current moment matrix
			idx_list = generate_unsorted_idx_list(M=i,d=self.d)
			idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
			
			# Find maximum index entry in current moment matrix
			max_index = [sum(x) for x in zip(idx_list[-1],idx_list[-1])]
			
			if max_index in self.idx_list:
				# Allocate space for moment matrix
				self.mm_mj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))
				self.mm_bj.append(cp.Variable((len(idx_list),len(idx_list)), PSD=True))
			else:
				break 



	def create_local_matrix(self):
		self.lm_mj = []
		self.lm_bj = []
		self.lm_bj2 = []

		num_constraint_polys = len(self.polynomial_coeffs_m)

		for j in range(num_constraint_polys):
			# Calculate the max number of localizing matrices
			for i in range(len(self.mm_mj)):

				# Generate list of moment indexes for q alpha
				idx_list = generate_unsorted_idx_list(M=self.q_alpha,d=self.d)
				idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))

				# Get largest index from the corresponding moment matrix
				key = str(i)+'_'+str(self.mm_mj[i].shape[0]-1)+'_'+str(self.mm_mj[i].shape[1]-1)
				mm_idx = self.mm_dict[key]
				
				# Find maximum index entry in current localizing matrix
				max_index = [sum(x) for x in zip(mm_idx,idx_list[-1])]

				if max_index in self.idx_list:
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
				
				# print(cur_local_matrix)
				# print(self.lm_mj[0].shape)
				# print(self.lm_mj[cur_local_matrix].shape)
				# print(' ')

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




	# Martingale constraints for a 5 dimensional turtlebot system
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
			f = copy.deepcopy(self.idx_list[n])	# Get the test function

			# For each test function perform the following derivatives:
			# dx, dy, dg, dcos(g), dsin(g)
			dx = mono_derivative(f,[0])
			dy = mono_derivative(f,[1])
			dg = mono_derivative(f,[2])
			dcosg = mono_derivative(f,[3])
			dsing = mono_derivative(f,[4])

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
			if dx[0] != 0:
				# generator term: v(S)*cos(g)*df/dx
				dx[1][COSG_IDX] += 1
				constriant_str += 'self.input_v()*dx[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'
			if dy[0] != 0:
				# generator term: v(S)*sin(g)*df/dy
				dy[1][SING_IDX] += 1
				constriant_str += 'self.input_v()*dy[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dy[1])]+'
			if dg[0] != 0:
				# generator term: w(S)*df/dg
				constriant_str += 'self.input_w()*dg[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dg[1])]+'
			if dcosg[0] != 0:
				# generator term: -sin(g)*df/dcos(g)
				dcosg[1][SING_IDX] += 1
				constriant_str += '-1*dcosg[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dcosg[1])]+'
			if dsing[0] != 0:
				# generator term: cos(g)*df/dsin(g)
				dsing[1][COSG_IDX] += 1
				constriant_str += 'dsing[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dsing[1])]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			exec(command)


		################ Degree 2 and above ###############################################
		# Get position in index list where monomial degree is 2
		start = next(i for i,v in enumerate(self.idx_list) if np.sum(self.idx_list[i])==2)
		
		# n represents the index of the monomial test function
		for n in range(start, len(self.mj)):
			f = copy.deepcopy(self.idx_list[n])	# Get the test function
			
			# For each test function perform the following derivatives:
			# dx, dy, dg, dcos(g), dsin(g), dx^2, dxdy, dydx, dy^2, dg^2  
			dx = mono_derivative(f,[0])
			dy = mono_derivative(f,[1])
			dg = mono_derivative(f,[2])
			dcosg = mono_derivative(f,[3])
			dsing = mono_derivative(f,[4])
			dx_2 = mono_derivative(f,[0,0])
			dxdy = mono_derivative(f,[0,1])
			dydx = mono_derivative(f,[1,0])
			dy_2 = mono_derivative(f,[1,1])
			dg_2 = mono_derivative(f,[2,2])
			# print('f',f)
			# print('dx',dx)
			# print('dy',dy)
			# print('dg',dg)
			# print('dt',dt)
			# print('dcosg',dcosg)
			# print('dsing',dsing)
			# print('dx_2',dx_2)
			# print('dxdy',dxdy)
			# print('dydx',dydx)
			# print('dy_2',dy_2)
			# print('dg_2',dg_2)

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
			if dx[0] != 0:
				# generator term: v(S)*cos(g)*df/dx
				dx[1][COSG_IDX] += 1
				constriant_str += 'self.input_v()*dx[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx[1])]+'
			if dy[0] != 0:
				# generator term: v(S)*sin(g)*df/dy
				dy[1][SING_IDX] += 1
				constriant_str += 'self.input_v()*dy[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dy[1])]+'
			if dg[0] != 0:
				# generator term: w(S)*df/dg
				constriant_str += 'self.input_w()*dg[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dg[1])]+'
			if dcosg[0] != 0:
				# generator term: -sin(g)*df/dcos(g)
				dcosg[1][SING_IDX] += 1
				constriant_str += '-1*dcosg[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dcosg[1])]+'
			if dsing[0] != 0:
				# generator term: cos(g)*df/dsin(g)
				dsing[1][COSG_IDX] += 1
				constriant_str += 'dsing[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dsing[1])]+'
			if dx_2[0] != 0:
				# generator term: 0.5*cos(g)^2*d^2f/dx^2
				dx_2[1][COSG_IDX] += 2
				constriant_str += '0.5*dx_2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dx_2[1])]+'
			if dxdy[0] != 0:
				# generator term: 0.5*cos(g)*sin(g)*d^2f/dxdy
				dxdy[1][COSG_IDX] += 1
				dxdy[1][SING_IDX] += 1
				constriant_str += '0.5*dxdy[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dxdy[1])]+'
			if dydx[0] != 0:
				# generator term: 0.5*cos(g)*sin(g)*d^2f/dydx
				dydx[1][COSG_IDX] += 1
				dydx[1][SING_IDX] += 1
				constriant_str += '0.5*dydx[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dydx[1])]+'
			if dy_2[0] != 0:
				# generator term: 0.5*sin(g)^2*d^2f/dy^2
				dy_2[1][SING_IDX] += 2
				constriant_str += '0.5*dy_2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dy_2[1])]+'
			if dg_2[0] != 0:
				# generator term: d^2f/dg^2
				constriant_str += 'dg_2[0]*'
				constriant_str += 'self.mj[self.idx_list.index(dg_2[1])]+'

			constriant_str = constriant_str[:-1] 	# Remove last '+' sign
			constriant_str += '==0'
			command = 'constraints.append('+constriant_str+')'
			exec(command)

		print('Num martingale constraints:', len(constraints))
		return constraints

	# Velocity control input v(S)
	def input_v(self):
		return 0.1

	# Angular velocity control input w(S) (omega)
	def input_w(self):
		return 0.1




if __name__ == '__main__':

	# idx_list = generate_unsorted_idx_list(M=0,d=6)
	# idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	# for i in idx_list:
	# 	print(i)
	# print(len(idx_list), '\n')


	# Notes: 
	# 1) q_alpha is the highest degree of polynomials q that define the safe set
	#    the size of pcs_m and pcs_b need to match the number of monomial terms for q_alpha
	# 
	# 2) State space dimensionality: 5
	#    state = [x, y, gamma (g), cos(g), sin(g)]

	state_dim = 5
	moment_degree = 5

	# Generate polynomial inequalities that define safe/unsafe sets
	# Square: x >= 0, -x+10 >= 0, y >= 0, -y+10 >= 0
	# q_polynomial_deg = 1
	# num_inequalities = 4
	# pcs_m0 = [0,1,0,0,0,0,0]
	# pcs_m1 = [10,-1,0,0,0,0,0]
	# pcs_m2 = [0,0,1,0,0,0,0]
	# pcs_m3 = [10,0,-1,0,0,0,0]
	# pcs_m = [pcs_m0,pcs_m1,pcs_m2,pcs_m3]
	# pcs_b = [[-i for i in pcs_m0],[-i for i in pcs_m1],[-i for i in pcs_m2],[-i for i in pcs_m3]]

	# Circle: x^2-10x+y^2-10y+50 <= 9
	#	      -> -41+10x+10y-x^2-y^2 >= 0
	q_polynomial_deg = 2
	num_inequalities = 1
	idx_list = generate_unsorted_idx_list(M=q_polynomial_deg,d=state_dim)
	idx_list = sorted(idx_list, key=cmp_to_key(index_comparison))
	pcs_m = np.zeros(len(idx_list)).tolist()
	pcs_m[idx_list.index([0,0,0,0,0])] = -41
	pcs_m[idx_list.index([1,0,0,0,0])] = 10
	pcs_m[idx_list.index([0,1,0,0,0])] = 10
	pcs_m[idx_list.index([2,0,0,0,0])] = -1
	pcs_m[idx_list.index([0,2,0,0,0])] = -1
	pcs_b = [[-i if i != 0 else 0 for i in pcs_m]]
	pcs_m = [pcs_m]


	# x=0.5, y=0.5, g=0.0, cos(g)=cos(1), sin(g)=sin(1)
	starting_state = [5,5,1.0,math.cos(1),math.sin(1)]

	ets = escape_time_solver(d=state_dim, M=moment_degree, q_alpha=q_polynomial_deg, 
		                     pcs_m=pcs_m, pcs_b=pcs_b, lm_vis=False, y0=starting_state,
		                     max_opt_iters=200000, solver_verbose=True)
	start_time = time.time()
	ets.solve(mode='Maximize')
	print('end time:',time.time()-start_time)

	pickle.dump(ets, open('results/5d_system/5000_iter/M4_max.pkl', 'wb'))
	