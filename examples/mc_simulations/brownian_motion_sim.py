import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import copy
import pickle
import time
import random


# Constants for indexing state variables
Y_IDX = 0
T_IDX = 1

def safe_set(state):

	if state[Y_IDX] <= 1.0 and state[Y_IDX] >= 0. and state[T_IDX] <= 10.:
		return True
	else:
		return False






class brownian_motion_mc:

	def __init__(self, dt, safe_set_func, init_state):

		# Sim functions
		self.safe_set_func = safe_set_func

		# Sim variables
		self.init_state = init_state
		self.state = copy.copy(init_state)
		self.t = 0.0
		self.cur_step = 0

		# Env params
		self.dt = dt


	def bm_increment(self):
		return norm.rvs(scale=self.dt**0.5)

	def reset(self):
		self.state = copy.copy(self.init_state)
		self.t = 0.0
		self.cur_step = 0

	def within_safe_set(self):
		return self.safe_set_func(self.state)

	def step(self):
		dB = self.bm_increment()
		d_y = dB
	
		self.state[Y_IDX] += d_y
		self.state[T_IDX] += self.dt

		self.t += self.dt
		self.cur_step += 1

		return self.within_safe_set()




if __name__ == '__main__':

	# State space: [y, t]
	# System:
	#	dy = dB
	# 	dt = dt


	# Initial conditions:
	starting_state = [0.1, 0.0]

	sim = brownian_motion_mc(dt=0.001, 
			     safe_set_func=safe_set, 
			     init_state=starting_state)



	# # Single Run
	# position = []
	# velocity = []
	# time = []
	# sim.reset()
	# for i in range(10000):
	# 	safe = sim.step()
	# 	if safe != True:
	# 		break
	# 	position.append(sim.state[POS_IDX])
	# 	velocity.append(sim.state[VEL_IDX])
	# 	time.append(sim.t)


	# print(np.max(position), np.min(position))
	# print(np.max(velocity), np.min(velocity))
		
	# plt.plot(time, position)
	# plt.grid()
	# plt.show()
	# plt.clf()
	# plt.plot(time, velocity)
	# plt.grid()
	# plt.show()




	# Multi Run
	start_time = time.time()
	random.seed(100)
	exit_times = []
	exit_locations = []
	num_runs = 200000
	cur_completed_percent = 0

	for run in range(num_runs):
		if int(run/num_runs*100) > cur_completed_percent:
			print('Progress: '+str(int(run/num_runs*100))+'%')
			cur_completed_percent = int(run/num_runs*100)

		sim.reset()
		for i in range(10000000):
			safe = sim.step()
			if not safe:
				break
		
		exit_times.append(sim.state[T_IDX])
		exit_locations.append(sim.state[Y_IDX])

	print('RUNTIME: ', time.time()-start_time)
	#pickle.dump([exit_times, exit_locations], open('SMD/exit_times/seed100_300000_jan30_ellipseSS.pkl', 'wb'))

	plt.hist(exit_times)
	print('First moment mean:', np.mean(exit_times), ' variance: ', np.var(exit_times))
	m2 = np.power(exit_times, 2)
	print('Second moment mean: ', np.mean(m2), 'variance: ', np.var(m2))
	plt.show()

















	# Need ~10 terms for taylor polynomial to appoximate cos(x)
	# x_list = np.linspace(0,8,10000)
	# t_list = np.linspace(0,8,10000)
	# for i in range(len(x_list)):
	# 	x = x_list[i]
	# 	s = 0
	# 	for n in range(10):
	# 		s += (-1)**n*(x**(2*n))/math.factorial(2*n)
	# 		print((-1)**n/math.factorial(2*n))
	# 	print(' ')
	# 	x_list[i] = s
	# 	t_list[i] = math.cos(t_list[i])

	# plt.plot(np.linspace(0,8,10000), x_list)
	# plt.plot(np.linspace(0,8,10000), t_list, color='red')
	# plt.show()


