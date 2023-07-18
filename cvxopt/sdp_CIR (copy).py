import cvxpy as cp 
import numpy as np 


# Notes:
# LM matrix defined in two ways rigth now (the supposidly error way shown in the paper,
# along with the "correct" way using mm0 
# Currently using a single polynimal q represented with r=0.1 (the region of safety)


def is_psd(x):
	return np.all(np.linalg.eigvals(x)>=0)


class escape_time_solver:
	def __init__(self, d, M, pcs_m, y0=0.2, alpha=1.0, beta=0.0, sigma=0.5, r=0.1):
		self.d = d
		self.M = M
		self.mj = [cp.Variable() for i in range(M+1)]
		self.mj_tilde = [cp.Variable() for i in range(M+2)]
		self.bj = [cp.Variable() for i in range(M+1)]
		self.mu_1 = cp.Variable()
		self.mu_r = cp.Variable()

		self.polynomial_coeffs_m = pcs_m

		# Moment matrices
		self.mm_mj = None
		self.mm_bj = None

		# Localizing matrices
		self.lm_mj = None
		self.lm_bj = None

		self.y0 = y0
		self.alpha = alpha
		self.beta = beta
		self.sigma = sigma
		self.r = r



	def solve(self):
		self.create_moment_matrix()
		self.create_local_matrix()

		constraints = []
		constraints += [self.bj[0] == self.mu_1 + self.mu_r]
		constraints += self.mj_tilde_constraints()
		constraints += self.mm_constraints()
		constraints += self.lm_constraints()
		constraints += self.martingale_constraint_CoV()
		#constraints += self.martingale_constraint()


		obj = cp.Maximize(self.mj[0])

		prob = cp.Problem(obj, constraints)
		prob.solve()

		print('MJs')
		for mj in self.mj:
			print(mj.value)
		print('\nMJ Tildes')
		for mjt in self.mj_tilde:
			print(mjt.value)
		print('\nBJs')
		for bj in self.bj:
			print(bj.value)

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


	def mj_tilde_constraints(self):
		# Link each moment mj with each change of measure moment mj_tilde
		# mj = (1-r)*mj+1_tilde + r*mj_tilde

		constraints = []
		for i in range(len(self.mj)):
			constraints.append(self.mj[i] == (1-self.r)*self.mj_tilde[i+1] + self.r*self.mj_tilde[i])
			
		return constraints


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


	def martingale_constraint_CoV(self):

		constraints = []

		# Monomial test function constraints
		for n in range(2, len(self.mj_tilde)-1):
			constraints.append(n*self.alpha*(1-self.r)*self.mj_tilde[n+1] 
							   + (2*n*self.alpha*self.r + ((n*(n-1)*self.sigma**2)/2) + n*self.beta)*self.mj_tilde[n]
							   + ((n*(n-1)*self.sigma**2*self.r + n*self.alpha*self.r**2 + n*self.beta*self.r)/(1-self.r))*self.mj_tilde[n-1]
							   + ((n*(n-1)*self.sigma**2*self.r**2)/(2*(1-self.r)**2))*self.mj_tilde[n-2]
							   + ((self.y0-self.r)/(1-self.r))**n
							   - self.mu_1
							   == 0)


		# Natural log test function constraint
		constraints.append((self.alpha*self.r + self.beta - self.sigma**2/2)*self.mj_tilde[0]
						   + self.alpha*(1-self.r)*self.mj_tilde[1]
						   + np.log(self.y0)
						   - np.log(self.r)*self.mu_r
						   == 0)

		return constraints

	def martingale_constraint(self):

		constraints = []

		# k=0 and k=1 constraints
		constraints.append(self.bj[0] - 1 == 0)
		constraints.append(self.bj[1] - self.y0 - self.alpha*self.mj[1] - self.beta*self.mj[0] == 0)

		# The following is dependent on this specific generator
		# Generator: y*f'(y) + 0.125y*f''(y)
		for k in range(2,self.M+1):

			d1_deriv_c = k  		# coefficient of first deriviative
			d1_deriv_d = k-1 		# degree of first derivative

			d2_deriv_c = k*(k-1) 	# coefficient of second deriviative
			d2_deriv_d = k-2		# degree of second derivative

			d1_moment_idx = d1_deriv_d+1
			d1_moment_coeff = d1_deriv_c*self.alpha

			d2_moment_idx = d2_deriv_d
			d2_moment_coeff = d2_deriv_c*((self.sigma**2)/2)

			constraints.append(self.bj[k] - self.y0**k
				               - d1_moment_coeff*self.mj[d1_moment_idx]
				               - d1_deriv_c*self.beta*self.mj[d1_deriv_d]
				               - d2_moment_coeff*self.mj[d2_moment_idx]
				               == 0)

		return constraints





if __name__ == '__main__':
	#ets = escape_time_solver(d=1, M=12, pcs_m=[-0.1,1.1,-1.0], pcs_b=[0.1,-1.1,1.0], y0=0.2)
	#ets.solve()
	#beta = [0.0,0.2,0.5,1.0,2.0,5.0]
	beta = [1.0,2.0,5.0]
	for b in beta:
		ets = escape_time_solver(d=1, M=12, pcs_m=[-0.1,1.1,-1.0], y0=0.2, beta=b)
	#ets = escape_time_solver(d=1, M=4, pcs_m=[0,1,-1], y0=0.2)
		ets.solve()
	#[-0.1,1.1,-1]
	#(x-0.1)(x-1) = x^2-1.1x+0.1
	#(x-1)(x+1) x^2 + 1
	#(x-1)x = x^2-x