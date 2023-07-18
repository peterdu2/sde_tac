import pickle
import numpy as np 
import cvxpy as cp 
from escape_time_suspension_higher_moments import escape_time_solver
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative


ets_deg1 = pickle.load(open('results/SMD/time_aug/M968_300000_min_deg1.pkl', 'rb'))
#ets_deg2 = pickle.load(open('results/SMD/time_aug/M150_3000000_max_deg2.pkl', 'rb'))


for i in range(5):
	print(i)
	d1 = ets_deg1.mj[ets_deg1.idx_list.index([0,0,i])].value
	#d2 = ets_deg2.mj[ets_deg1.idx_list.index([0,0,i])].value
	print(d1)
	#print(d2)
	#print(d2/d1, '\n')

# for i in range(len(ets_deg1.mj)):
# 	d1 = ets_deg1.mj[i].value
# 	d2 = ets_deg2.mj[i].value
# 	print(d1)
# 	print(d2)
# 	print(d2/d1, '\n')


