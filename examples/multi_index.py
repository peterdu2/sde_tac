import numpy as np
from functools import cmp_to_key
import copy

# https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


# Input:
# 	M: Maximum degree of index entries
#   d: Dimensionality of polynomial
# Output:
#   List of unsorted multi-indicies of d variables up to degree M 

def generate_unsorted_idx_list(M, d):
	idx_list = [[0 for x in range(d)]]
	for cur_degree in range(0,M+1):

		# Generate all permutations of dimension d
		L = list(sums(d,cur_degree))
		
		for idx in L:
			idx_entry = [x for x in idx]
			if idx_entry not in idx_list:
				idx_list.append(idx_entry)
		
	return idx_list



# Input:
#	Two multi indicies represetned as lists
# Output:
#	True if item1 < item2 

def index_comparison(item1, item2):

	# If deg(item1) < deg(item2) then item1 < item2
	# If deg(item1) == deg(item2):
	#	The larger element is the one whose exponent vector (multi-index) is lexically smaller.
	#	From left to right, first element that differs, the vector with higher exponent
	#	is the one smaller lexically
	# 	Ref: https://people.sc.fsu.edu/~jburkardt/m_src/polynomial/polynomial.html

	if sum(item1) != sum(item2):
		return -1 if sum(item1) < sum(item2) else 1

	for i in range(0,len(item1)):
		if item1[i] != item2[i]:
			return -1 if item1[i] > item2[i] else 1
				


# Function to take the derivative of monomials
# Input:
#	List of degrees representing a monomial Eg: x^2y^4 = [2,4]
#	List of variables to take derivatives with respect to Eg: df/dxdy = [0,1], df/dx^2 = [0,0]
# Output:
#	List of degrees representing a monomial after differentiation
#	Value of coefficient after differentiation 
#	Eg: if the function after diff. is 6x^2y^3, return value is [6, [2,3]]

def mono_derivative(monomial, dx_list):

	coeff = 1
	degrees = copy.copy(monomial)

	for deriv_var in dx_list:

		if degrees[deriv_var] == 0:	# Check if taking derivative will result in zero
			return [0, []]
		else:
			coeff *= degrees[deriv_var]
			degrees[deriv_var] -= 1

	return [coeff, degrees]



# print(mono_derivative([3,0,0], [1,1]))
# print(mono_derivative([3,1,0], [1]))
# print(mono_derivative([3,2,0], [0,1]))
# print(mono_derivative([0,0,0], [2]))
