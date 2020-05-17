
import os
import ctypes
import cplex
import math
import numpy as np
import time

from test_cplex_solver import *
from scipy import linalg
from ctypes import *
from itertools import product
from numpy.ctypeslib import ndpointer

INF = 1E30

TOL_1 = 1E-16
TOL_2 = 1E-12
TOL_3 = 1E-8
TOL_4 = 1E-4
TOL_5 = 1E2

EPSILON_P = 10E-7
EPSILON_r = 10E-9
EPSILON_D = 10E-7
EPSILON_a = 10E-6
EPSILON_z = 10E-12
EPSILON_0 = 10E-14

CODE_PHASE_I_BASIS_DUAL_FEASIBLE = 4
CODE_PHASE_I_PROBLEM_DUAL_UNFEASIBLE = 5
CODE_PROBLEM_OPTIMAL_BASIS = 6
CODE_PROBLEM_DUAL_UNBOUNDED = 7


# comando per creare una dll: gcc -shared -o perf_test.so -fPIC -O2 perf_test.c
clib = ctypes.cdll.LoadLibrary(
	"/home/macs/coding/optimization/cuplex/test/clib.so")

# int get_row(double * rhs, int nrows, int * it, double tol, double inf);
clib.get_row.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),c_int, POINTER(c_int), c_double, c_double]
clib.get_row.restype = c_int

# int get_col(double * mat, double * obj, int * fSet, int ncols, int it, int * jt, double tol, double inf);
clib.get_col.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),ndpointer(c_double, flags="C_CONTIGUOUS"),ndpointer(c_int, flags="C_CONTIGUOUS"),c_int, c_int, POINTER(c_int), c_double, c_double]
clib.get_col.restype = c_int

# double pivot(double *mat, double *obj, double *rhs, int nrows, int ncols, int it, int jt, double currObj, double tol)
clib.pivot.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),
					   ndpointer(c_double, flags="C_CONTIGUOUS"),
					   ndpointer(c_double, flags="C_CONTIGUOUS"),
					   c_int, c_int, c_int, c_int, POINTER(c_double), c_double]
clib.pivot.restypes = c_int


class Variable(object):
	"""
	Represent a problem variable.

	vtype: continuous, binary, integer
	"""

	def __init__(self, obj, lb=0.0, ub=INF, vtype="C"):
		self.obj = obj
		self.lb = lb
		self.ub = ub
		self.vtype = vtype


def read_input_file(problem_file_path):

	# txt file
	row_cnt = -1
	col_cnt = 0
	cnt = 0
	obj_list = []

	curr_row = []
	handler = open(problem_file_path, "r")

	m, n = [int(i) for i in handler.readline().split()]
	table = np.zeros((m, n), dtype=c_double)
	obj = np.zeros(n, dtype=c_double)

	for line in handler:
		spt = [int(i) for i in line.split()]

		if len(obj_list) < n:
			obj_list += spt

		elif len(spt) == 1 and cnt == col_cnt:

			if row_cnt > -1:
				for i in curr_row:
					table[row_cnt, i - 1] = 1

			row_cnt += 1
			col_cnt = spt[0]
			cnt = 0

			curr_row = []
		else:
			cnt += len(spt)
			curr_row += spt

	for i in curr_row:
		table[row_cnt, i - 1] = 1

	for j in range(n):
		obj[j] = obj_list[j]
	handler.close()

	return table, obj


def rev_prim_simplex(mat, obj, lb, ub):
	opt_flag = False
	unb_flag = False

	m, n = mat.shape

	f_set = np.ones(ncols, dtype=c_int)
	base = np.array(range(m, ncols), dtype=c_int)

	for j in range(m, ncols):
		f_set[j] = 0

	tab = np.hstack((-mat.T, np.eye(n, dtype=c_double)))
	obj = np.hstack((np.ones(m), np.zeros(n, dtype=c_double)))
	rhs = -np.array(c, dtype=c_double)

	Y = obj[base].dot(np.linalg.inv(tab[:,base]))

	z = 0
	z_prev = 0
	itercnt = 0
	zunchcnt = 0
	while (not opt_flag) and (not unb_flag):
		b_inv = np.linalg.inv(tab[:, base])
		delta_a = obj[base].dot(b_inv).dot(tab) - obj

		# FIND COL
		opt_flag = True
		minimum = INF
		jt = -1
		for j in range(ncols):
			if (delta_a[j] <= -tol) and (delta_a[j] < minimum):
				minimum = delta_a[j]
				jt = j
			opt_flag = opt_flag and (delta_a[j] > -tol)

		if opt_flag:
			print("Optimal solution found!")
			break

		# FIND ROW
		ba_jt = b_inv.dot(tab[:,jt])
		bb = b_inv.dot(rhs)

		minimum = INF
		it = -1
		for i in range(nrows):
			if (ba_jt[i] > tol) and (bb[i] / ba_jt[i] < minimum):
				minimum = (bb[i] / ba_jt[i])
				it = i

		z -= rhs[it] * obj[jt]

		# UPDATE
		f_set[base[it]] = 1
		f_set[jt] = 0
		base[it] = jt

		print("iteration=%d, z=%lf" % (itercnt, z))
		itercnt += 1

	u = obj[n:]

	print("Objective value=%8.4lf, iterations=%5d" % (z, itercnt))
	return u, c - u.dot(mat), z

	return 0


def phase_I_dual_simplex(mat, c, m, n, tol, log_freq):
	pass


def phase_I_pans_method(mat, obj, lb, ub, verbosity_level):

	this_func_name = "phase_I_pans_method"

	m, n = mat.shape

	# select a start basis
	base = np.array(range(n-m,n), dtype=c_int)
	while np.linalg.matrix_rank(mat[:,base]) != m:
		base = np.array(np.random.choice(range(n), m, False), dtype=c_int)

	# PAN'S METHOD: Algorithm 11 pag 48(64)
	# initialization
	B_inv = np.linalg.inv(mat[:,base])
	y = B_inv.dot(obj[base])

	d = np.zeros(n, dtype=c_double)
	for j in range(n):
		if j not in base:
			d[j] = obj[j] - mat[:,j].dot(y)

	exit_code = -1
	while True:

		# SELECT ENTERING VARIABLE Q
		# pag 38(54) e 40(56)
		M = set()
		P = set()
		maximum = 0
		q = -1
		for j in range(n):

			# J_u and M
			if (abs(lb[j] + INF) < TOL_5) and (ub[j] < INF) and d[j] > 0:
				M.add(j)
				if abs(d[j]) > maximum:
					maximum = abs(d[j])
					q = j

			# J_l and P
			if (abs(lb[j] + INF) < TOL_5) and (lb[j] > -INF) and d[j] < 0:
				P.add(j)
				if abs(d[j]) > maximum:
					maximum = abs(d[j])
					q = j

		Q = M.union(P)
		if len(Q) == 0:
			exit_code = CODE_PHASE_I_BASIS_DUAL_FEASIBLE
			if verbosity_level > 0:
				print(this_func_name + ": Basis B is dual feasible")
			break

		# FTran
		alpha_q = B_inv.dot(mat[:, q])

		# Vedere Algorthm 9 pag 43(59)

		# SELECT LEAVING VARIABLE
		H = set()
		r = -1
		maximum = 0
		for i in range(m):
			if (lb[base[i]] > -INF and alpha_q[i] < TOL_1) or (ub[base[i]] < INF and alpha_q[i] > TOL_1):
				H.add(i)
				if abs(alpha_q[i]) > maximum:
					maximum = abs(alpha_q[i])
					r = i

		if len(H) == 0:
			exit_code = CODE_PHASE_I_PROBLEM_DUAL_UNFEASIBLE
			if verbosity_level > 0:
				print(this_func_name + ": Problem is dual unfeasible")
			break

		p = base[r]
		#rho_r = B_inv[r]
		alpha_r = mat.T.dot(B_inv[r])
		theta_D = d[q] / alpha_r[q]

		# BASIS CHANGE AND UPDATE
		for j in range(n):
			if j not in base and j != q:
				d[j] = d[j] - theta_D * alpha_r[j]

		d[q] = 0
		d[p] = -theta_D

		base[r] = q
		B_inv = np.linalg.inv(mat[:,base])

	return exit_code, base



def lu_update_forrest_tomlin(l, u, alpha):
	"""
	See paragraph 5.3.1 pages 57(73) to 59(75) of 'The Dual Simplex Method,
	Techniques for a fast and stable implementation' by Achim Koberstein

	https://d-nb.info/978580478/34
	"""

	return l, u


def lu_update_suhl(l, u, r, t, alpha, P, P_inv):
	"""
	See Algorithm 13 page 62(78) of 'The Dual Simplex Method,
	Techniques for a fast and stable implementation' by Achim Koberstein

	https://d-nb.info/978580478/34
	"""

	m = u.shape[0]

	# insert vector alpha into column r of U
	u[:,r] = alpha

	for k in range(t+1, l):
		i = P[k]
		if abs(u[i,r]) > TOL_1:
			l[r,i] = - ur[r,i] / u[i,i]
			u[r,i] = 0
			for j in range(m):
				if j != i:
					tmp = u[r,j]
					u[r,j] = u[r,j] + L[r,i] * u[i,j]
					
	for k in range(t+1, l):
		i = P_inv[k]
		P[i] = k - 1
		P_inv[k-1] = i

	return l, u, P, P_inv



def dual_simplex_elab(mat, obj, lb, ub, base, verbosity_level):

	this_func_name = "dual_simplex_elab"
	use_lu_decomposition = False
	use_simple_ratio_test = True

	m, n = mat.shape

	base_set = set(base)
	non_base = np.array(list(set(range(n)).difference(base_set)), dtype=c_int)

	# set all primal varibles (basic and non-basic) to a finite bound
	# in a successive step, set basic variables as x := B^-1 * N * x_N 
	x = np.zeros(n, dtype=c_double)
	for j in range(n):
		x[j] = lb[j] if abs(lb[j]) < abs(ub[j]) else ub[j]

	# see, algorithm 7, page 35 - simplex thesis
	# INITIALISATION
	
	###########################################################
	if not use_lu_decomposition:
		B_inv = np.linalg.inv(mat[:,base])
	else:
		lu_perm, l_B, u_B = linalg.lu(mat[:,base])								# lu decomposition
		l_B_inv = linalg.inv(l_B)
		u_B_inv = linalg.inv(u_B)

		B_inv = np.linalg.inv(mat[:, base])
		assert np.isclose(B_inv, l_B_inv.dot(u_B_inv)).all()
	###########################################################

	lu_perm = np.array(range(m), dtype=c_int)
	lu_perm_inv = np.array(range(m), dtype=c_int)
	
	b_tilde = -mat[:,non_base].dot(x[non_base])
	

	###########################################################
	if not use_lu_decomposition:
		x[base] = B_inv.dot(b_tilde)
		y = B_inv.T.dot(obj[base])
	else:
		x[base] = l_B_inv.dot(u_B_inv.dot(b_tilde))							# lu decomposition
		y = u_B_inv.T.dot(l_B_inv.T.dot(obj[base]))

		assert np.isclose(B_inv.dot(b_tilde), x[base]).all()
		assert np.isclose(B_inv.T.dot(obj[base]), y).all()
	###########################################################


	d_non_base = obj[non_base] - mat[:,non_base].T.dot(y)
	d = np.hstack((obj[non_base], np.zeros(m, dtype=c_double))) - np.hstack((mat[:,non_base], np.zeros((m,m), dtype=c_double))).T.dot(y)
	beta = np.ones(m, dtype=c_double)

	z = obj.dot(x)
	itercnt = 0
	exit_code = -1
	while True:

		###############################		PRICING		#################################
		p = -1
		r = -1
		maximum = -INF
		opt_flag = True
		for i in range(m):
			ii = base[i]

			# see 3.52
			delta = (x[ii] - lb[ii]) if (x[ii] < lb[ii]) else ((x[ii] - ub[ii]) if (x[ii] > ub[ii]) else 0)
			
			# see 3.53
			if (abs(delta) > EPSILON_0) and ((delta ** 2 / beta[i]) > maximum):
				maximum = (delta ** 2 / beta[i])
				p = ii
				r = i

			opt_flag = opt_flag and (lb[ii] <= x[ii] <= ub[ii])

		#delta = (x[p] - lb[p]) if (x[p] < lb[p]) else (x[p] - ub[p])
		delta = (x[p] - lb[p]) if (x[p] < lb[p]) else ((x[p] - ub[p]) if (x[p] > ub[p]) else 0)
		
		#print(delta, x[p], ub[p], lb[p])

		if opt_flag:
			exit_code = CODE_PROBLEM_OPTIMAL_BASIS
			if verbosity_level > 0:
				print(this_func_name + ": Optimal basis found")
			break

		# BTran e Pivot Row
		# per rho_r e e_r vedi pag 18(34) - se p è la variabile che lascia la base corrente B
		# r è la colonna della base
		
		e_r = np.zeros(m, dtype=c_double)
		e_r[r] = 1.0

		###########################################################
		if not use_lu_decomposition:
			rho_r = B_inv.T.dot(e_r)
		else:
			rho_r = l_B_inv.T.dot(u_B_inv.T.dot(e_r))						# lu decomposition

			assert np.isclose(linalg.inv(mat[:,base]).T.dot(e_r), rho_r).all()
		###########################################################

		#alpha_r = mat[:,non_base].T.dot(rho_r)
		alpha_r = mat.T.dot(rho_r)

		###############################		RATIO TEST		#############################
		# vedere Algorithm 4 pag 31(47)
		
		if x[p] < lb[p]:
			alpha_tilde_r = -alpha_r
			delta_0 = lb[p] - x[p]
		if x[p] > ub[p]:
			alpha_tilde_r = alpha_r
			delta_0 = x[p] - ub[p]

		if use_simple_ratio_test:
			# Simple Ratio Test - from Algorithm 2 pag 26(42)
			F = set()
			for j in range(n):
				# TODO: aggiungere free variables!!
				if (j not in base_set) and (((abs(x[j] - lb[j]) < EPSILON_0) and (alpha_tilde_r[j] > EPSILON_a)) or ((abs(x[j] - ub[j]) < EPSILON_0) and (alpha_tilde_r[j] < -EPSILON_a))):
					F.add(j)

			if len(F) == 0:
				exit_code = CODE_PROBLEM_DUAL_UNBOUNDED
				if verbosity_level > 0:
					print(this_func_name + ": Problem is dual unbounded")
				break

			theta = INF
			q = -1
			for j in F:
				tmp = d[j] / alpha_tilde_r[j]
				if (tmp < theta) or ((tmp < (theta + EPSILON_z)) and (abs(alpha_tilde_r[j]) > abs(alpha_tilde_r[q]))):
					theta = tmp
					q = j
		else:
			
			# # Selection of q with the BRFT
			# delta = abs(delta)
			# Q = set()
			# for j in range(n):
			# 	# TODO: aggiungere free variable
			# 	if (j not in base_set) and (((abs(x[j] - lb[j]) < EPSILON_0) and (alpha_tilde_r[j] > EPSILON_0)) or ((abs(x[j] - ub[j]) < EPSILON_0) and (alpha_tilde_r[j] < -EPSILON_0))):
			# 		Q.add(j)
			
			# print("lenQ=", len(Q), "delta=", delta)
			# q = -1
			# prev_q = -1
			# while (len(Q) != 0) and (delta > -EPSILON_0):
			# 	minimum = INF
			# 	for j in Q:
			# 		if (d[j] / alpha_tilde_r[j]) < minimum:
			# 			minimum = (d[j] / alpha_tilde_r[j])
						
			# 			if prev_q == -1:
			# 				prev_q = j
			# 				q = j
			# 			else:
			# 				prev_q = q
			# 				q = j

			# 	delta = delta - (ub[q] - lb[q]) * abs(alpha_r[q])
			# 	Q.discard(q)
			# #print("lenQ=", len(Q), "delta=", delta)
			
			# if len(Q) == 0:
			# 	exit_code = CODE_PROBLEM_DUAL_UNBOUNDED
			# 	if verbosity_level > 0:
			# 		print(this_func_name + ": Problem is dual unbounded")
			# 	break
			
			# if delta < -EPSILON_0:
			# 	q = prev_q

			# BFRT with Harri's tolerance
			# See algorithm 18 pag 78(94)

			# phase I
			Q = set()
			val1 = INF
			val2 = INF
			for j in range(n):
				if (j not in base_set):
					if (abs(x[j] - lb[j]) < EPSILON_0) and (alpha_tilde_r[j] > 0):
						Q.add(j)
						val1 = min(val1, (d[j] + EPSILON_D) / alpha_tilde_r[j])

					elif (abs(x[j] - ub[j]) < EPSILON_0) and (alpha_tilde_r[j] < 0):
						Q.add(j)
						val2 = min(val2, (d[j] - EPSILON_D) / alpha_tilde_r[j])

			THETA_MAX = min(val1, val2)

			# phase II
			theta_D = 10 * THETA_MAX
			delta = delta_0
			delta_cap = 0
			while (delta - delta_cap) >= 0:
				delta = delta - delta_cap
				Q_tilde = set()
				val1 = INF
				val2 = INF
				for j in Q:
					# Q_l
					if (alpha_tilde_r[j] > 0):
						if ((d[j] - theta_D*alpha_tilde_r[j]) < -EPSILON_D):
							Q_tilde.add(j)
						val1 = min(val1, (d[j] + EPSILON_D) / alpha_tilde_r[j])

					# Q_u
					elif (alpha_tilde_r[j] < 0):
						if ((d[j] - theta_D*alpha_tilde_r[j]) > EPSILON_D):
							Q_tilde.add(j)
						val2 = min(val2, (d[j] - EPSILON_D) / alpha_tilde_r[j])
				THETA_MAX = min(val1, val2)

				Q.difference_update(Q_tilde)
				delta_cap = 0
				for j in Q_tilde:
					delta_cap += (ub[j] - lb[j]) * abs(alpha_tilde_r[j])
	
				theta_D = 2 * THETA_MAX

			# phase III
			while (len(Q_tilde) != 0) and (delta >= 0):
				val1 = INF
				val2 = INF
				for j in Q_tilde:
					# Q_tilde_l
					if (alpha_tilde_r[j] > 0):
						val1 = min(val1, (d[j] + EPSILON_D) / alpha_tilde_r[j])

					# Q_tilde_u
					elif (alpha_tilde_r[j] < 0):
						val2 = min(val2, (d[j] - EPSILON_D) / alpha_tilde_r[j])

				THETA_MAX = min(val1, val2)
				K = set()
				maximum = -INF
				val = 0
				q = -1
				for j in Q_tilde:
					if (d[j] / alpha_tilde_r[j]) <= THETA_MAX:
						K.add(j)
						val += (ub[j] - lb[j]) * abs(alpha_tilde_r[j])
						if abs(alpha_tilde_r[j]) > maximum:
							maximum = abs(alpha_tilde_r[j])
							q = j

				Q_tilde.difference_update(K)
				delta -= val

			if len(Q_tilde) == 0:
				exit_code = CODE_PROBLEM_DUAL_UNBOUNDED
				if verbosity_level > 0:
					print(this_func_name + ": Problem is dual unbounded")
				break

			
		#print("q=", q, "d[q]=", d[q], "alpha_r[q]=",alpha_r[q], "alpha_tilde_r[q]=", alpha_tilde_r[q])
		theta_D = d[q] / alpha_r[q]


		###########################################################
		if not use_lu_decomposition:
			# FTran
			alpha_q = B_inv.dot(mat[:,q])

			# DSE FTran
			tau = B_inv.dot(rho_r)
		else:
			# FTran
			alpha_q = u_B_inv.dot(l_B_inv.dot(mat[:,q]))					# lu decomposition

			# DSE FTran
			tau = u_B_inv.dot(l_B_inv.dot(rho_r))

			assert np.isclose(linalg.inv(mat[:, base]).dot(mat[:, q]), alpha_q).all()
			assert np.isclose(linalg.inv(mat[:, base]).dot(rho_r), tau).all()
		###########################################################

		err_q = abs(d[q] - (obj[q] - obj[base].dot(alpha_q)))
		if not (err_q < EPSILON_0):
			if verbosity_level > 0:
				print("Recuded cost test failed: err=%lf" % (err_q))
			break

		# BASIS CHANGE and UPDATE
		# Vedere Algorithm 5 pag 31(47)
		T = set()
		delta_z = 0
		a_tilde = np.zeros(m, dtype=c_double)

		for j in range(n):

			if j in base_set:
				continue

			d[j] = d[j] - theta_D * alpha_r[j]
			if not (abs(ub[j] - lb[j]) < EPSILON_0):
				if (abs(x[j] - lb[j]) < EPSILON_0) and (d[j] < -EPSILON_0):
					T.add(j)
					a_tilde = a_tilde + (ub[j] - lb[j]) * mat[:,j]
					delta_z = delta_z + (ub[j] - lb[j]) * obj[j]
				elif (abs(x[j] - ub[j]) < EPSILON_0) and (d[j] > EPSILON_0):
					T.add(j)
					a_tilde = a_tilde + (lb[j] - ub[j]) * mat[:,j]
					delta_z = delta_z + (lb[j] - ub[j]) * obj[j]

		d[p] = -theta_D
		if len(T) != 0:
			
			###########################################################
			if not use_lu_decomposition:
				delta_x_base = B_inv.dot(a_tilde)
			else:
				delta_x_base = u_B_inv.dot(l_B_inv.dot(a_tilde))
			###########################################################
			
			x[base] = x[base] - delta_x_base
			delta_z = delta_z - sum([obj[base[i]] * delta_x_base[i] for i in range(m)])

		z = z + delta_z

		theta_p = delta / alpha_q[r]
		x[base] = x[base] - theta_p * alpha_q[:]
		x[q] = x[q] + theta_p

		# Update β by formulas (3.47a) and (3.50) - pag 32(48)
		beta_r = beta[r]
		for i in range(m):
			if i == r:
				beta[i] = (1 / alpha_q[r]) ** beta_r
			else:
				beta[i] = beta[i] - 2 * (alpha_q[i] / alpha_q[r]) * tau[i] + (alpha_q[i] / alpha_q[r]) ** 2 * beta_r

		#print(base[r], p)
		base_set.discard(p)
		base_set.add(q)
		base[r] = q
		
		###########################################################
		if not use_lu_decomposition:
			B_inv = np.linalg.inv(mat[:,base])
		else:
			l_B, u_B = linalg.lu(mat[:, base], True)					# lu decomposition
			l_B_inv = linalg.inv(l_B)
			u_B_inv = linalg.inv(u_B)

			assert np.isclose(linalg.inv(mat[:, base]), u_B_inv.dot(l_B_inv)).all()
		###########################################################

		# Flip bounds in xN by algorithm 6 -  pag 31(47)
		for j in T:
			if abs(x[j] - lb[j]) < EPSILON_0:
				x[j] = ub[j]
			else:
				x[j] = lb[j]

		z = z + theta_D * delta


		####		PRIMAL AND DUAL ERROR CHECK			####
		#err_prim = max(np.abs(b_tilde - mat[:,base].dot(x[base])))
		#err_dual = max(np.abs(obj[base] - mat[:,base].T.dot(y)))
		#print("prim error: %lf" % (err_prim))
		#print("dual error: %lf" % (err_dual))

		itercnt += 1
		#print("z=%8.4lf, iter=%5d" % (z, itercnt))

	if verbosity_level > 0:
		print("Objective value=%8.4lf, iterations=%5d" % (z, itercnt))
	return exit_code, z, itercnt


def dual_simplex(mat, c, m, n, tol, log_freq):
	opt_flag = False
	unb_flag = False

	# phase I
	nrows = m
	ncols = n + m

	f_set = np.ones(ncols, dtype=c_int)
	base = np.array(range(n, n + m), dtype=c_int)

	for j in range(n, ncols):
		f_set[j] = 0

	tab = np.hstack((-mat, np.eye(m, dtype=c_double)))
	obj = np.hstack((c, np.zeros(m, dtype=c_double)))
	rhs = -np.ones(m, dtype=c_double)

	z = 0
	z_prev = 0
	itercnt = 0
	zunchcnt = 0
	while (not opt_flag) and (not unb_flag):
		
		# FIND ROW
		opt_flag = True
		maximum = INF
		it = -1
		for i in range(nrows):
			if (rhs[i] <= -tol) and (rhs[i] < maximum):
				maximum = rhs[i]
				it = i
			opt_flag = opt_flag and (rhs[i] > -tol)
		
		# it_ = c_int(-1)
		# res = clib.get_row(rhs, c_int(nrows), byref(it_), c_double(tol), c_double(INF))
		# if it != it_.value:
		# 	print("iter=%4d - it1=%4d != it2=%4d" % (itercnt, it, it_.value))
		# if opt_flag != res:
		# 	print("iter=%4d - opt1=%4d != opt2=%4d" % (itercnt, opt_flag, res))

		if opt_flag:
			print("Optimal solution found!")
			break
		
		# FIND COL
		minimum = INF
		jt = -1
		for j in range(ncols):
			if (tab[it, j] < -tol) and (obj[j] / abs(tab[it, j]) < minimum) and f_set[j]:
				minimum = (obj[j] / abs(tab[it, j]))
				jt = j

		# jt_ = c_int(-1)
		# res = clib.get_col(tab, obj, f_set, c_int(ncols), c_int(it), byref(jt_), c_double(tol), c_double(INF))
		# if jt != jt_.value:
		# 	print("iter=%4d - jt1=%4d != jt2=%4d" % (itercnt, jt, jt_.value))

		if jt == -1:
			unb_flag = True
			print("Unbounded solution found")
			break
		


		# PIVOT
		# tab_cp = tab.copy()
		# obj_cp = obj.copy()
		# rhs_cp = rhs.copy()

		tmp = tab[it, jt]
		tab[it] = tab[it] / tmp
		rhs[it] = rhs[it] / tmp

		for i in range(nrows):
			if (abs(tab[i, jt]) > tol) and (i != it):
				tmp = -tab[i, jt]
				tab[i] += (tab[it] * tmp)
				rhs[i] += (rhs[it] * tmp)
		
		z_cp = c_double(z)
		z -= rhs[it] * obj[jt]
		obj -= tab[it] * obj[jt]

		# clib.pivot(tab_cp, obj_cp, rhs_cp, c_int(nrows), c_int(ncols), c_int(it), c_int(jt), byref(z_cp), c_double(tol))
		# if not np.isclose(tab, tab_cp).all():
		# 	print("iter=%4d - tab != tab_cp" % (itercnt))
		# if not np.isclose(obj, obj_cp).all():
		# 	print("iter=%4d - obj != obj_cp" % (itercnt))
		# if not np.isclose(rhs, rhs_cp).all():
		# 	print("iter=%4d - rhs != rhs_cp" % (itercnt))
		# if not np.isclose(z, z_cp.value):
		# 	print("iter=%4d - z1=%8.4f != z2=%8.4f" % (itercnt, z, z_cp.value))

		# UPDATE
		f_set[base[it]] = 1
		f_set[jt] = 0
		base[it] = jt

		itercnt += 1
	
	u = obj[n:]

	print("Objective value=%8.4lf, iterations=%5d" % (z, itercnt))
	return u, c - u.dot(mat), z


instcnt = 0
instcnt_passed = 0

verbosity_level = 0

for i in range(50):
	inst = "_demo%02d" % (i)

	if verbosity_level > 0:
		print("\ninst:", inst)
	mat, obj = read_input_file("/home/macs/coding/optimization/cuplex/data/scp" + inst + ".txt")

	cplex_val, cplex_iter = solve_with_cplex(mat.copy(), obj.copy())

	m, n = mat.shape

	# TRANSFORM TO COMPUTATIONAL FORM, with rhs = 0
	mat = np.hstack((mat, np.eye(m, dtype=c_double)))
	obj = np.hstack((obj, np.zeros(m, dtype=c_double)))
	lb = np.hstack((np.zeros(n, dtype=c_double), -INF * np.ones(m, dtype=c_double)))
	ub = np.hstack((np.ones(n, dtype=c_double), -np.ones(m, dtype=c_double)))

	code, base = phase_I_pans_method(mat.copy(), obj.copy(), lb.copy(), ub.copy(), verbosity_level)
	if code == CODE_PHASE_I_PROBLEM_DUAL_UNFEASIBLE:
		continue

	try:
		code, z, my_iter = dual_simplex_elab(mat.copy(), obj.copy(), lb.copy(), ub.copy(), base.copy(), verbosity_level)
		
		if verbosity_level > 0:
			print("cplex = %8.4lf, my = %8.4lf - cplex iter = %4d, my iter = %4d" % (cplex_val, z, cplex_iter, my_iter))

		if abs(cplex_val - z) < 10E-10:
			instcnt_passed += 1
	except:
		if verbosity_level > 0:
			print("Something wrong")

	instcnt += 1

print("\nTotal instances = %d, passed = %d" % (instcnt, instcnt_passed))
