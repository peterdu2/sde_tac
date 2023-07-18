import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import copy
import pickle
import time
import random


# Constants for indexing state variables
POS_IDX = 0
VEL_IDX = 1
TIME_IDX = 2

def safe_set(state):
	# Without noise: max = 2.002218577692498
	#				 min = 1.927225430031328
	# Set position safe set at [1.2, 2.65]
	# Set velocity safe set at limit at [-2, 2]
	# if state[POS_IDX] > 2.65 or state[POS_IDX] < 1.2:
	# 	return False
	# return True

	# ellipse shape: -3.180005 + 3.85x1 - x1^2 - 1/9x2^2 >= 0
	# if -3.180005 + 3.85*state[POS_IDX] - state[POS_IDX]**2 - 1/9*state[VEL_IDX]**2 >= 0 and state[TIME_IDX] <= 15:
	# 	return True
	# else:
	# 	return False

	# # parabola 1 shape: -3.18 + 3.85x1 - x1^2 >= 0
	# if -3.18 + 3.85*state[POS_IDX] - state[POS_IDX]**2 >= 0:
	# 	return True
	# else:
	# 	return False

	# return True
	# # parabola 2 shape:  -2.35 + 3.35x1 -x1^2 >= 0
	# if -2.35 + 3.35*state[POS_IDX] - state[POS_IDX]**2 >= 0 and state[TIME_IDX] <= 15:
	# 	return True
	# else:
	# 	return False

	if state[POS_IDX] < -1.0 and state[POS_IDX] > -2.3 and state[TIME_IDX] <= 50:
		return True
	else:
		return False






class SMD_mc:

	def __init__(self, dt, safe_set_func, init_state=[0.5,0.5,0.0], params=[0.1,0.1,1]):

		# Sim functions
		self.safe_set_func = safe_set_func

		# Sim variables
		self.init_state = init_state
		self.state = copy.copy(init_state)
		self.t = 0.0
		self.cur_step = 0

		# Env params
		self.dt = dt
		self.g = 9.81
		self.ks = params[0]
		self.ms = params[1]
		self.kc = params[2]


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
		d_x1 = (self.state[VEL_IDX])*self.dt
		d_x2 = (-(self.ks/self.ms)*self.state[POS_IDX] - self.g + (self.kc/self.ms)*self.state[POS_IDX]*self.state[VEL_IDX])*self.dt	# Deterministic term
		d_x2 += (self.kc/self.ms)*dB #Noise term
		#d_x2 += dB
	
		self.state[POS_IDX] += d_x1
		self.state[VEL_IDX] += d_x2

		self.state[TIME_IDX] += self.dt
		self.t += self.dt
		self.cur_step += 1

		return self.within_safe_set()




if __name__ == '__main__':

	# State space: [position, velocity]
	# Control inputs: Reactive damper force Fd(S)

	# Parameters
	ks = 5.
	ms = 1
	kc = 1.

	# Initial conditions:
	# position = 9.81/ks, velocity = 0.3, time = 0
	starting_state = [-ms*9.81/ks, 0.0, 0.0]

	sim = SMD_mc(dt=0.005, 
			     safe_set_func=safe_set, 
			     params=[ks,ms,kc],
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
	num_runs = 1000
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
	#pickle.dump([exit_times, exit_locations], open('SMD/exit_times/seed100_300000_jan30_ellipseSS.pkl', 'wb'))

	plt.hist(exit_times)
	print('First moment', np.mean(exit_times), np.var(exit_times))
	m2 = np.power(exit_times,2)
	print('Second moment', np.mean(m2), np.var(m2))
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


