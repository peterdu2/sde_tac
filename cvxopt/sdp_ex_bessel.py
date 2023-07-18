import cvxpy as cp 
import numpy as np 


# Notes:
# LM matrix defined in two ways rigth now (the supposidly error way shown in the paper,
# along with the "correct" way using mm0 
# Currently using a single polynimal q represented with r=0.1 (the region of safety)


def is_psd(x):
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:
	def __init__(self, M, pcs_m, y0, r, d):
		self.d = d
		self.M = M
		self.mj = [cp.Variable() for i in range(M+1)]
		self.bj = [cp.Variable() for i in range(M+1)]
		self.mu_1 = cp.Variable()
		self.mu_r = cp.Variable()

		self.polynomial_coeffs_m = pcs_m
		self.y0 = y0

		# Moment matrices
		self.mm_mj = None
		self.mm_bj = None

		# Localizing matrices
		self.lm_mj = None
		self.lm_bj = None

		self.alpha = (d-1)/2
		self.r = r



	def solve(self):
		self.create_moment_matrix()
		self.create_local_matrix()

		constraints = []
		constraints += [self.bj[0] == self.mu_1 + self.mu_r]
		constraints += self.mm_constraints()
		constraints += self.lm_constraints()
		constraints += self.martingale_constraint_bessel()


		obj = cp.Maximize(self.mj[0])

		prob = cp.Problem(obj, constraints)
		prob.solve()

		# print('MJs')
		# for mj in self.mj:
		# 	print(mj.value)
		# print('\nBJs')
		# for bj in self.bj:
		# 	print(bj.value)

		print(' ')

		print(prob.value)
		print(prob.status)





	def create_moment_matrix(self):
		# Calculate the max number of moment matrices
		# Case when d=1, max # MM = floor(M/2) since last entry of MM
		# has index equal to double the dimension
		num_mms = self.M//2+1 

		# Create a list of matrix variables (each entry is a moment matrix)
		self.mm_mj = [cp.Variable((i,i), PSD=True) for i in range(1,num_mms+1)]
		self.mm_bj = [cp.Variable((i,i), PSD=True) for i in range(1,num_mms+1)]


	def create_local_matrix(self):

		num_mms = self.M//2+1

		for num_lm in range(num_mms,0,-1):
			max_mm_idx = 2*(self.mm_mj[num_lm-1].shape[0]-1) # largest index in moment matrix
			max_alpha = len(self.polynomial_coeffs_m)-1	# largest alpha (index of polynomial coeffs)
			max_lm_idx = max_mm_idx + max_alpha		# largest index in localising matrix
			if max_lm_idx <= len(self.mj)-1:		# check if largest lm index is within moment arrays 
				break

		self.lm_mj = [cp.Variable((i,i), PSD=True) for i in range(1,num_lm+1)]
		self.lm_bj = [cp.Variable((i,i), PSD=True) for i in range(1,num_lm+1)]



	def mm_constraints(self):
		# Generate a list of constraints that connect moment matrix to moment array
		num_mms = self.M//2+1
		constraints = []

		for cur_mat in range(num_mms):
			# Link to moments constraints
			for i in range(cur_mat+1):
				for j in range(cur_mat+1):
					moment_idx = i+j
					constraints.append(self.mm_mj[cur_mat][i][j] == self.mj[moment_idx])
					constraints.append(self.mm_bj[cur_mat][i][j] == self.bj[moment_idx])

		return constraints


	def lm_constraints(self):
		constraints = []

		for cur_mat in range(len(self.lm_mj)):
			# Link to moments constraints
			for i in range(cur_mat+1):
				for j in range(cur_mat+1):
					mm_idx = i+j

					constraints.append(self.lm_mj[cur_mat][i][j] 
									   == self.polynomial_coeffs_m[0]*self.mj[mm_idx]		#alpha=0
									   + self.polynomial_coeffs_m[1]*self.mj[mm_idx+1]	#alpha=1
									   + self.polynomial_coeffs_m[2]*self.mj[mm_idx+2])	#alpha=2

					constraints.append(self.lm_bj[cur_mat][i][j] 
									   == -self.polynomial_coeffs_m[0]*self.bj[mm_idx]		#alpha=0
									   + -self.polynomial_coeffs_m[1]*self.bj[mm_idx+1]	#alpha=1
									   + -self.polynomial_coeffs_m[2]*self.bj[mm_idx+2])	#alpha=2

					constraints.append(self.lm_bj[cur_mat][i][j] 
									   == self.polynomial_coeffs_m[0]*self.bj[mm_idx]		#alpha=0
									   + self.polynomial_coeffs_m[1]*self.bj[mm_idx+1]	#alpha=1
									   + self.polynomial_coeffs_m[2]*self.bj[mm_idx+2])	#alpha=2

		return constraints



	def martingale_constraint_bessel(self):

		constraints = []

		# k=0 and k=1 constraints
		constraints.append(self.mu_1 - 1 == 0)
		constraints.append(self.alpha*self.mj[1]
						   + (self.r/(1-self.r))*self.alpha*self.mj[0]
						   + ((self.y0-self.r)/(1-self.r))**2
						   - self.mu_1
						   == 0)

		for n in range(2, self.M+1):
			# Monomial constraint
			constraints.append(((n*(n-1)/2) + n*self.alpha)*self.mj[n]
							   + n*(self.r/(1-self.r))*(self.alpha+n-1)*self.mj[n-1]
							   + (n*(n-1)/2)*(self.r*(1-self.r))**2*self.mj[n-2]
							   + ((self.y0-self.r)/(1-self.r))**2
							   - self.mu_1
							   == 0)


		# Natural log constraint
		constraints.append((self.alpha-0.5)*self.mj[0]
			               + np.log(self.y0)
			               + np.log(self.r)*self.mu_r
			               == 0)

		return constraints




if __name__ == '__main__':
	#ets = escape_time_solver(d=1, M=12, pcs_m=[-0.1,1.1,-1.0], pcs_b=[0.1,-1.1,1.0], y0=0.2)
	#ets.solve()
		#ets = escape_time_solver(d=1, M=12, pcs_m=[-0.1,1.1,-1.0], pcs_b=[0.1,-1.1,1.0], y0=0.2)
	ets = escape_time_solver(M=12, pcs_m=[0,1,-1], y0=0.4, r=0.1, d=10)
	ets.solve()
	#[-0.1,1.1,-1]
	#(x-0.1)(x-1) = x^2-1.1x+0.1
	#(x-1)(x+1) x^2 + 1
	#(x-1)x = x^2-x