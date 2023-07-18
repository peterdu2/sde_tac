import pickle
import numpy as np 
import matplotlib.pyplot as plt


data = pickle.load(open('OU_process/p_safety/safer_p_values/seed100_params3.pkl', 'rb'))

num_safe = data[0]
num_unsafe = data[1]
params = data[3]
ss = data[2]
time_upper = data[6]
print(num_unsafe, num_safe)
print(num_unsafe/(num_unsafe+num_safe))
print(params)
print(ss)
print(time_upper)
# m2 = np.power(data,2)
# #plt.hist(data, bins=1000)
# # plt.hist(m2, bins=70)
# # plt.show()
# print(np.mean(m2))
# print(np.var(m2))
