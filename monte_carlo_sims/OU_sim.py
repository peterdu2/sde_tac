import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import copy
import pickle
import time
import random


# Simulate an Ornstein Uhlenbeck Process (aka Langevin Equation)
# dXt = -aXt dt + b dBt

POS_IDX = 0
TIME_IDX = 1


def safe_set(state, ss_lower, ss_upper, t_max):
	if state[POS_IDX] < ss_lower or state[POS_IDX] > ss_upper or state[TIME_IDX] > t_max:
		return False
	return True







class OU_sim:

	def __init__(self, dt, safe_set_func, init_state=[0.0, 0.0], params=[1.0, 1.0],
				 safe_set_lower=0.0, safe_set_upper=10.0, time_upper=100.0):

		# Sim functions
		self.safe_set_func = safe_set_func

		# Sim variables
		self.init_state = init_state
		self.state = copy.copy(init_state)
		self.t = 0.0
		self.cur_step = 0

		# Env params
		self.dt = dt
		self.alpha = params[0]
		self.sigma = params[1]
		self.ss_lower = safe_set_lower
		self.ss_upper = safe_set_upper
		self.time_upper = time_upper


	def bm_increment(self):
		return norm.rvs(scale=self.dt**0.5)

	def reset(self):
		self.state = copy.copy(self.init_state)
		self.t = 0.0
		self.cur_step = 0


	def within_safe_set(self):
		return self.safe_set_func(self.state, self.ss_lower, self.ss_upper, self.time_upper)

	def step(self):
		dB = self.bm_increment()
		#dB = np.sqrt(self.dt)*np.random.randn()
		
		dx = -self.alpha * self.state[POS_IDX] * self.dt
		dx += self.sigma * dB 
		
		self.state[POS_IDX] += dx
		self.state[TIME_IDX] += self.dt
		self.t += self.dt
		self.cur_step += 1

		return self.within_safe_set()




if __name__ == '__main__':

	# Initial conditions:
	# position = 0.0, time = 0.0
	starting_state = [1.05, 0.0]
	starting_state = [9.6, 0.0]
	#starting_state = [.5, 0.0]
	
	# System parameters:
	alpha = 0.525
	sigma = 0.688
	params = [alpha, sigma]
	#params = [alpha, sigma]
	
	# Safe set params
	safe_set_lower = 1.
	safe_set_upper = 10.
	time_upper = 30.

	sim = OU_sim(dt=0.005, 
			     safe_set_func=safe_set, 
			     init_state=starting_state,
				 params=params,
				 safe_set_lower=safe_set_lower,
				 safe_set_upper=safe_set_upper,
				 time_upper=time_upper)


	# # Multi Run
	# # p-safety simulator
	# start_time = time.time()
	# random.seed(100)
	# num_safe = 0
	# num_unsafe = 0
	# num_runs = 50000
	# cur_completed_percent = 0

	# for run in range(num_runs):
	# 	if int(run/num_runs*100) > cur_completed_percent:
	# 		print('Progress: '+str(int(run/num_runs*100))+'%')
	# 		cur_completed_percent = int(run/num_runs*100)

	# 	sim.reset()
	# 	for i in range(1000000):
	# 		safe = sim.step()
	# 		# if safe and sim.state[TIME_IDX] == time_upper:
	# 		# 	num_safe += 1
	# 		# 	break
	# 		# elif not safe:
	# 		# 	num_unsafe += 1
	# 		# 	break
	# 		if not safe and sim.state[TIME_IDX] >= time_upper:
	# 			num_safe += 1
	# 			break
	# 		elif not safe and sim.state[TIME_IDX] < time_upper:
	# 			num_unsafe += 1
	# 			#print(sim.state)
	# 			break
	

	# print('RUNTIME: ', time.time()-start_time)
	# pickle.dump([num_safe, num_unsafe, starting_state, params, safe_set_lower, safe_set_upper, time_upper, num_runs], open('OU_process/p_safety/safer_p_values/seed100_params1.pkl', 'wb'))

	# print(num_safe, num_unsafe, num_unsafe/(num_unsafe+num_safe))




	# Single Run
	# position = []
	# time = []
	# sim.reset()
	# for i in range(10000):
	# 	safe = sim.step()
	# 	if safe != True:
	# 		print(sim.state[POS_IDX])
	# 		print(sim.t)
	# 		break
	# 	position.append(sim.state[POS_IDX])
	# 	time.append(sim.t)


	# print(np.max(position), np.min(position))
	
	# print(position)
	# plt.plot(time, position)
	# plt.grid()
	# plt.show()




	# Multi Run
	# Exit Time simulator
	start_time = time.time()
	random.seed(100)
	exit_times = []
	exit_locations = []
	num_runs = 50000
	cur_completed_percent = 0

	for run in range(num_runs):
		if int(run/num_runs*100) > cur_completed_percent:
			print('Progress: '+str(int(run/num_runs*100))+'%')
			cur_completed_percent = int(run/num_runs*100)

		sim.reset()
		for i in range(1000000):
			safe = sim.step()
			if not safe:
				break
		
		exit_times.append(sim.state[TIME_IDX])
		exit_locations.append(sim.state[POS_IDX])

	print('RUNTIME: ', time.time()-start_time)
	pickle.dump([exit_times, exit_locations, starting_state, params, safe_set_lower, safe_set_upper, time_upper], open('OU_process/p_safety/safer_p_values/seed100_params1_exit_times_50k.pkl', 'wb'))

	plt.hist(exit_times, bins=50)
	print(np.mean(exit_times), np.var(exit_times))
	print(np.mean(np.power(exit_times,2)))
	plt.show()
	plt.savefig('OU_process/p_safety/safer_p_values/exit_times_hist_params1.png')

















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


