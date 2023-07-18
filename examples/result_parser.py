import cvxpy as cp
import numpy as np
import time
from functools import cmp_to_key
from multi_index import generate_unsorted_idx_list, index_comparison, mono_derivative
from et_sus_sinusoidal import escape_time_solver
#from et_sus_old import escape_time_solver
import copy
import math
import pickle

if __name__ == '__main__':
    # ets = pickle.load(open('results/SpringMassDamper_M6.pkl', 'rb'))
    # moment_matrix = np.zeros((ets.moment_matrix_mj.shape[0], ets.moment_matrix_mj.shape[1]))
    # for i in range(moment_matrix.shape[0]):
    #     for j in range(moment_matrix.shape[1]):
    #         index = ets.mm_dict[str(i)+'_'+str(j)]
    #         moment_val = ets.mj[ets.idx_list.index(index)].value
    #         moment_matrix[i][j] = moment_val

    # np.set_printoptions(precision=1)
    # print('Moment Matrix:\n', moment_matrix, '\n')

    # mm_eigvals, mm_eigvecs = np.linalg.eig(moment_matrix)
    # print(mm_eigvals.shape[0], 'eigenvalues:\n', mm_eigvals, '\n')

    # cond_n = np.linalg.cond(moment_matrix)
    # print('Condition number of moment matrix:', cond_n, '\n')

    # lm_mj_p1 = np.array(ets.localising_matrices_mj[0].value)
    # lm_mj_p2 = np.array(ets.localising_matrices_mj[1].value)
    # print('Condition number of MJ localising matrices:', np.linalg.cond(lm_mj_p1), ',', np.linalg.cond(lm_mj_p2), '\n')

    # lm_bj_p1 = np.array(ets.localising_matrices_bj[0].value)
    # lm_bj_p2 = np.array(ets.localising_matrices_bj[1].value)
    # print('Condition number of BJ localising matrices:', np.linalg.cond(lm_bj_p1), ',', np.linalg.cond(lm_bj_p2), '\n')

    # ets = pickle.load(open('results/9_28.pkl', 'rb'))

    # mm_mj = np.array(ets.mm_mj[-1].value)
    # np.set_printoptions(precision=1)
    # print('Moment Matrix:\n', mm_mj, '\n')

    # mm_eigvals, mm_eigvecs = np.linalg.eig(mm_mj)
    # print(mm_eigvals.shape[0], 'eigenvalues:\n', mm_eigvals, '\n')

    # cond_n = np.linalg.cond(mm_mj)
    # print('Condition number of moment matrix:', cond_n, '\n')

    # lm_bj_p1 = np.array(ets.lm_bj_inner[0][-1].value)
    # print('Condition number of BJ localising matrices:', np.linalg.cond(lm_bj_p1), '\n')

    # print(lm_bj_p1)

    # values_min = pickle.load(open('results/brownian_motion_min.pkl', 'rb'))
    # values_max = pickle.load(open('results/brownian_motion_max.pkl', 'rb'))
    analytical = [0.250000,
                  0.104167,
                  0.063542,
                  0.051525,
                  0.052208,
                  0.063478,
                  0.090044,
                  0.145973]


    values_min = pickle.load(open('results/quadcopter_height_min.pkl', 'rb'))[0]
    values_max = pickle.load(open('results/quadcopter_height_max.pkl', 'rb'))[0]
    runtime_min = pickle.load(open('results/quadcopter_height_min.pkl', 'rb'))[1]
    runtime_max = pickle.load(open('results/quadcopter_height_max.pkl', 'rb'))[1]

    #print(len())
    for i in range(3):
        print('M'+str(i+1)+'', '   Lower Bound:', '{:.6f}'.format(values_min[i]), '   Runtime:', '{:.6f}'.format(runtime_min[i]), '   Upper Bound:', '{:.6f}'.format(
            values_max[i]), '   Runtime:', '{:.6f}'.format(runtime_max[i]), '   Analytical:', '{:.6f}'.format(analytical[i]))

    print(np.mean(runtime_min))
    print(np.mean(runtime_max))