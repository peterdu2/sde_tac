import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import copy
import pickle
import time


# Constants for indexing state variables
X_IDX = 0
Y_IDX = 1
G_IDX = 2
T_IDX = 3
COSG_IDX = 4
SING_IDX = 5



def safe_set(state):

	# Square:
	# if state[X_IDX]>=0 and -state[X_IDX]+10>=0 and state[Y_IDX]>=0 and -state[Y_IDX]+10>=0:
	# 	return True
	# else:
	# 	return False

	# Circle:
	# radius 3 circle centred at (x,y)=(5,5)
	# -41+10x+10y-x^2-y^2 >= 0
	if -41+10*state[X_IDX]+10*state[Y_IDX]-state[X_IDX]**2-state[Y_IDX]**2 >= 0:
		return True
	else:
		return False

def v_S(state):
	return 0.1

def w_S(state):
	return 0.1



class turtlebot_mc:

	def __init__(self, dt, safe_set_func, v_S, w_S, init_state=[0.5,0.5]):
		self.dt = dt
		self.safe_set_func = safe_set_func
		self.v_S = v_S
		self.w_S = w_S
		self.init_state = init_state
		self.state = copy.copy(init_state)
		self.t = 0.0

	def bm_increment(self):
		return norm.rvs(scale=self.dt**0.5)

	def reset(self):
		self.state = copy.copy(self.init_state)
		self.t = 0.0

	def get_v_S(self):
		return self.v_S(self.state)

	def get_w_S(self):
		return self.w_S(self.state)

	def within_safe_set(self):
		return self.safe_set_func(self.state)

	def step(self):
		dB = self.bm_increment()
		dB = np.sqrt(self.dt)*np.random.randn()
		dx = self.get_v_S()*math.cos(self.state[G_IDX])*self.dt + math.cos(self.state[G_IDX])*dB
		dy = self.get_v_S()*math.sin(self.state[G_IDX])*self.dt + math.sin(self.state[G_IDX])*dB
		dg = self.get_w_S()*self.dt

		#print(dB, dB2)
		#print(math.cos(self.state[G_IDX])*dB, self.get_v_S()*math.cos(self.state[G_IDX])*self.dt)
		self.state[X_IDX] += dx
		self.state[Y_IDX] += dy
		self.state[G_IDX] += dg 
		self.state[T_IDX] += self.dt
		# self.state[COSG_IDX] = math.cos(self.state[G_IDX])
		# self.state[SING_IDX] = math.sin(self.state[G_IDX])
		# self.t += self.dt

		# if self.state[G_IDX] > math.pi:
		# 	self.state[G_IDX] -= math.pi
		# elif self.state[G_IDX] < 0:
		# 	self.state[G_IDX] += math.pi

		return self.within_safe_set()




if __name__ == '__main__':

	# State space: [x,y,g,t,cos(g),sin(g)]
	# Control inputs: velocity input v(S), angular velocity input w(S)
	# Dynamics: dx = v(S)cos(g)dt + cos(g)dB1
	#			dy = v(S)sin(g)dt + sin(g)dB1
	#			dg = w(t)dt + dB2
	#			dt = dt
	#			dcos(g) = -sin(g)*dg
	#			dsin(g) = cos(g)*dg

	# Initial conditions:
	# x=5, y=5, g=0.0, t=0.0, cos(g)=cos(1), sin(g)=sin(1)
	starting_state = [5,5,1.0,0.0,math.cos(1),math.sin(1)]

	sim = turtlebot_mc(dt=0.01, 
					   safe_set_func=safe_set, 
					   v_S=v_S,
					   w_S=w_S,
					   init_state=starting_state)

	positions = [[],[]]
	exit_times = []
	exit_locations = [[],[]]
	num_runs = 5000
	cur_completed_percent = 0

	for run in range(num_runs):
		if int(run/num_runs*100) > cur_completed_percent:
			print('Progress: '+str(int(run/num_runs*100))+'%')
			cur_completed_percent = int(run/num_runs*100)

		sim.reset()
		for i in range(1000000):
			positions[0].append(sim.state[X_IDX])
			positions[1].append(sim.state[Y_IDX])
			safe = sim.step()
			if not safe:
				break
		
		exit_times.append(sim.state[T_IDX])
		exit_locations[X_IDX].append(sim.state[X_IDX])
		exit_locations[Y_IDX].append(sim.state[Y_IDX])

		# plt.plot(positions[0], positions[1])
		# x = np.linspace(2, 8, 100)
		# y = np.linspace(2, 8, 100)
		# X, Y = np.meshgrid(x,y)
		# F = (X-5)**2 + (Y-5)**2 - 9
		# plt.contour(X,Y,F,[0])
		# plt.show()
		# print(exit_times)
		# time.sleep(5)

	#pickle.dump([exit_times, exit_locations], open('exit_times/circle/et_100000_trial1.pkl', 'wb'))

	plt.hist(exit_times)
	print(np.mean(exit_times), np.var(exit_times))
	plt.show()
	plt.clf()
	plt.scatter(exit_locations[X_IDX], exit_locations[Y_IDX])
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


