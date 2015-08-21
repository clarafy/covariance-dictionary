""" Covariance dictionary learning
"""

import warnings
from numpy import (arange, cov, diag, diag_indices, dot, empty, fmax, 
	identity, logical_or, max, mod, sqrt, where, triu_indices, vstack, zeros)
from numpy.linalg import norm
from scipy.linalg import eigh

def proj_psd(A):

	# Projects a symmetric matrix to nearest
	# positive semi-definite matrix.

	# TODO: Check for symmetry? Should never need to.

	d, U = eigh(A, lower=False)
	U = U[:, d > 0]
	d = d[d > 0]
	Aproj = dot(dot(U, diag(d)), U.T)

	return Aproj

def proj_corr(A):

	# TODO: If ends up being useful, add max_iter and tol parameters.

	# Projects a symmetric matrix to the nearest correlation
	# matrix (PSD matrix with equality constraints on
	# the diagonal and bound constraints on the off-diagonal)
	# using Dykstra's algorithm, as described in Higham (2002)
	# "Computing the nearest correlation matrix: A problem from finance".

	# How exactly is Dykstra's different from ADMM for two projections?

	n = A.shape[0]
	deltaS = zeros((n, n))
	Y = A
	X = zeros((n, n)) 

	triu_idx = triu_indices(n)
	diag_idx = diag_indices(n) 

	for n_iter in range(1, 101):

		Xprev = X
		Yprev = Y

		R = Y - deltaS

		# Project onto semidefinite cone.
		X = proj_psd(R)

		deltaS = X - R
		Y = X

		# Equality and bound constraints. 
		Y[diag_idx] = 1
		Y[Y > 1] = 1
		Y[Y < -1] = -1

		diffX = max(abs(X - Xprev)) / max(abs(X))
		diffY = max(abs(Y - Yprev)) / max(abs(Y))
		diffXY = max(abs(Y - X)) / max(abs(Y))

		if max([diffX, diffY, diffXY]) < 1e-6:
			break

	return Y



def proj_col_psd(A, correlation=False):

	# Projects every column of a matrix to the upper-triangle
	# of the nearest positive semi-definite or correlation matrix.

	n_pair, n_col = A.shape
	n = int((sqrt(1 + 8 * n_pair) - 1) / 2)

	Aproj = empty((n_pair, n_col))
	cov_triu = empty((n, n))
	triu_idx = triu_indices(n)

	if correlation:
		for mod_idx in range(n_col):

			# Reconstruct symmetric matrix (only need half).
			cov_triu[triu_idx] = A[:, mod_idx]
			Aproj[:, mod_idx] = proj_corr(cov_triu)[triu_idx]
	else:
		for mod_idx in range(n_col):

			# Reconstruct symmetric matrix (only need half).
			cov_triu[triu_idx] = A[:, mod_idx]
			Aproj[:, mod_idx] = proj_psd(cov_triu)[triu_idx]

	return Aproj



class CovarianceDictionary(object):

	def __init__(self, init='kmeans', method='als', n_components=2, max_iter=200, 
		tol=1e-6, nls_max_iter=2000, psdls_max_iter=2000, nls_beta=0.9, psdls_beta=0.9,
		correlation=False, admm_gamma=1.6, admm_alpha=1e-5):
		
		if init not in ('kmeans', 'rand'):
			raise ValueError(
                'Invalid initialization: got %r instead of one of %r' %
                (init, ('kmeans', 'rand')))
		self.init = init

		if method not in ('als', 'admm'):
			raise ValueError(
                'Invalid method: got %r instead of one of %r' %
                (method, ('als', 'admm')))
		self.method = method

		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.nls_max_iter = nls_max_iter
		self.psdls_max_iter = psdls_max_iter
		self.nls_beta = nls_beta
		self.psdls_beta = psdls_beta
		self.correlation = correlation
		self.admm_gamma = admm_gamma
		self.admm_alpha = admm_alpha



	def _initialize(self, X):

		# Initializes the modules M and weights W randomly or using k-means,
		# as in Wild, Curry, & Dougherty (2004) "Improving non-negative 
		# matrix factorizations through structured initialization".

		from sklearn.cluster import KMeans

		n_pair, n_samp = X.shape
		n = int((sqrt(1 + 8 * n_pair) - 1) / 2)

		if self.init == 'kmeans':

			Xnorm = X / norm(X, axis=0)

			km = KMeans(n_clusters=self.n_components).fit(Xnorm.T)
			centroids = km.cluster_centers_.T
			Minit = proj_col_psd(centroids)

			labels = km.predict(Xnorm.T)
			Winit = zeros((self.n_components, n_samp))
			Winit[labels, arange(n_samp)] = 1

		elif self.init == 'rand':

			# Initialize modules to random linear combinations
			# of input covariances.
			Minit = dot(X, rand(n_samp, self.n_components))
			Winit, _, _ = _nls_subproblem(X, Minit, rand(k, n_samp))

		return Minit, Winit



	def _admm(self, X, Minit, Winit):

		# Solves for covariance module and weights using ADMM. Reformulate
		# original optimization problem

		# minimize ||X - MW||_F
		# subject to
		# each column of M is a PSD matrix
		# each element of W is non-negative

		# as

		# minimize ||X - MW||_F
		# subject to
		# M = U
		# W = V
		# each column of U is a PSD matrix
		# each element of V is non-negative

		# and sequentially minimize the augmented Lagrangian
		# w.r.t U, V, M, and W.

		n_pair, n_samp = X.shape
		n = int((sqrt(1 + 8 * n_pair) - 1) / 2)
		max_dim = max([n_pair, n_samp])

		M = Minit
		W = Winit
		V = Winit
		Lambda = zeros((n_pair, k))
		Pi = zeros((k, n_samp))

		normX = norm(X)
		objective = empty(self.max_iter + 1)
		objective[0] = finfo('d').max

		for n_iter in range(1, self.max_iter + 1):

			alpha = self.admm_alpha * normX * max_dim / n_iter
			beta = alpha * n_samp / n_pair

			U = dot(dot(X, V.T) + alpha * M - Lambda, inv(dot(V, V.T) + alpha * identity(k)))
			V = dot(inv(dot(U.T, U) + beta * identity(k)), dot(U.T, X) + beta * W - Pi)

			if self.correlation:
				M, _ = proj_col_corr(U + Lambda / alpha)
			else:
				M = proj_col_psd(U + Lambda / alpha)
			
			W = fmax(V + Pi / beta, 0)

			Lambda = Lambda + gamma * alpha * (U - M)
			Pi = Pi + gamma * beta * (V - W)

			obj = norm(X - dot(M, W)) / normX
			obj_prev = objective[n_iter - 1]

			if abs(obj - obj_prev) / fmax(1, obj_prev) < self.tol or obj < self.tol:
				objective[n_iter] = obj
				break

			objective[n_iter] = obj

			if mod(n_iter, 10) == 0:
				print 'Objective: %f.' % obj

		objective = objective[1 : n_iter + 1]

		return M, W, objective



	def _nls_subproblem(self, X, Minit, Winit):

		# Update weights by solving non-negative least-squares
		# using projected gradient descent (basically a transposed 
		# version of scikit-learn's NMF _nls_subproblem method).

		W = Winit
		MtX = dot(M.T, X)
		MtM = dot(M.T, M)

		alpha = 1

		for n_iter in range(1, self.nls_max_iter + 1):

			grad = dot(MtM, W) - MtX

			# Stopping condition on projected gradient norm.
			# Multiplication with a boolean array is more than twice
			# as fast as indexing into grad.
			if norm(grad * logical_or(grad < 0, W > 0)) < self.tol:
				break

			Wold = W

			# Search for step size that produces sufficient decrease
			# ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
			# condition in Lin (2007) Eq. (17)).
			for inner_iter in range(5):

				# Gradient step.
				Wnew = W - alpha * grad

				# Projection step.
				Wnew *= Wnew > 0

				# Check Lin (2007) Eq. (17) condition.
				d = Wnew - W
				gradd = dot(grad.ravel(), d.ravel())
				dQd = dot(dot(MtM, d).ravel(), d.ravel())
				suff_decr = 0.99 * gradd + 0.5 * dQd < 0

				# 1.1 If initially not sufficient decrease, then...
				# 2.1 If initially sufficient decrease, then...
				if inner_iter == 0:
					decr_alpha = not suff_decr

				if decr_alpha:
					# 1.3 ...there is sufficient decrease.
					if suff_decr:
						M = Mnew
						break
					# 1.2 ...decrease alpha until...
					else:
						alpha *= self.nls_beta

				# 2.3 ...there is not sufficient decrease.
				elif not suff_decr or (Mold == Mnew).all():
					M = Mold
					break

				# 2.2 ...increase alpha until...
				else:
					alpha /= self.nls_beta
					Mold = Mnew

		if n_iter == self.nls_max_iter:
			warnings.warn("Max iterations reached in NLS subproblem.")

		return W, grad, n_iter 
		

	def _psdls_subproblem(self, X, Minit, W):

		# Update modules by solving column-wise positive-semidefinite (PSD)
		# constrained least-squares using projected gradient descent:

		# minimize ||X - M * W||_F
		# subject to the constraint that every column of M
		# corresponds to the upper triangle of a PSD matrix.

		n_pair, n_samp = X.shape
		n = int((sqrt(1 + 8 * n_pair) - 1) / 2)

		M = Minit
		WWt = dot(W, W.T)
		XWt = dot(X, W.T)
		pg_norm = empty(self.max_iter)
		# in_iter = empty(self.max_iter)
		
		alpha = 1

		for n_iter in range(1, self.psdls_max_iter + 1):

			gradM = dot(M, WWt) - XWt

			# Stopping condition on projected gradient norm.
			pg_norm[n_iter - 1] = norm(proj_col_psd(M - gradM, n) - M)
			if pg_norm[n_iter - 1] < self.tol:
				break

			Mold = M

			# Search for step size that produces sufficient decrease
			# ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
			# condition in Lin (2007) Eq. (17).)
			for inner_iter in range(20):

				# Gradient and projection steps.
				Mnew = proj_col_psd(M - alpha * gradM, n)

				d = Mnew - M
				gradd = dot(gradM.ravel(), d.ravel())
				dQd = dot(dot(d, WWt).ravel(), d.ravel())
				suff_decr = 0.99 * gradd + 0.5 * dQd < 0

				# 1.1 If initially not sufficient decrease, then...
				# 2.1 If initially sufficient decrease, then...
				if inner_iter == 0:
					decr_alpha = not suff_decr

				if decr_alpha:
					# 1.3 ...there is sufficient decrease.
					if suff_decr:
						M = Mnew
						break
					# 1.2 ...decrease alpha until...
					else:
						alpha *= self.psdls_beta

				# 2.3 ...there is not sufficient decrease.
				elif not suff_decr or (Mold == Mnew).all():
					M = Mold
					break

				# 2.2 ...increase alpha until...
				else:
					alpha /= self.psdls_beta
					Mold = Mnew

			# in_iter[n_iter - 1] = inner_iter

		if n_iter == self.psdls_max_iter:
			warnings.warn("Max iterations reached in SDLS subproblem.")

		# pg_norm = pg_norm[: n_iter]
		# in_iter = in_iter[: n_iter]

		return M, gradM, n_iter


	def _als(self, X, Minit, Winit):

		# Solves for covariance module and weights using
		# alternating constrained least-squares. Same framework
		# as scikit-learn's ALS for NMF.

		n_pair, n_mat = X.shape
		n = int((sqrt(1 + 8 * n_pair) - 1) / 2)
		
		M = Minit
		W = Winit

		# Initial gradient.
		gradM = dot(M, dot(W, W.T)) - dot(X, W.T)
		gradW = dot(dot(M.T, M), W) - dot(M.T, X)
		init_grad_norm = norm(vstack((gradM, gradW.T)))
		print "Initial gradient norm: %f." % init_grad_norm

		# Stopping tolerances for constrained ALS subproblems.
		tolM = self.tol * init_grad_norm
		tolW = tolM

		normX = norm(X)
		objective = empty(self.max_iter)
		# pg_norm = empty(self.max_iter)

		for n_iter in range(1, self.max_iter + 1):

			# Stopping criterion, based on Calamai & More (1987) Lemma 3.1(c)
			# (stationary point iff projected gradient norm = 0).
			pgradW = gradW * logical_or(gradW < 0, W > 0)
			pgradM = proj_col_psd(M - gradM, n) - M
			pgn = norm(vstack((pgradM, pgradW.T)))
			# pg_norm[n_iter - 1] = pgn

			if pgn < self.tol * init_grad_norm:
				break

			if mod(n_iter, 10) == 0:
				print 'Iterations: %i. Projected gradient norm: %f.' % (n_iter, pgn)

			# Record objective.
			obj = norm(X - dot(M, W)) / normX
			objective[n_iter - 1] = obj

			# Update modules.
			M, gradM, iterM = self._psdls_subproblem(X, M, W)
			if iterM == 1:
				tolM = 0.1 * tolM

			# Update weights.
			W, gradW, iterW = self._nls_subproblem(X, M, W)
			if iterW == 1:
				tolH = 0.1 * tolW

		print 'Iterations: %i. Final projected gradient norm: %f.' % (n_iter, pgn)

		objective = objective[: n_iter - 1]
		# pg_norm = pg_norm[: n_iter - 1]

		return M, W, objective



	def fit_transform(self, X):
		
		"""Learns a covariance dictionary from data X and returns dictionary weights."""

		Minit, Winit = self._initialize(X)

		if self.method == 'als':
			M, W, obj = self._als(X, Minit, Winit)
		elif self.method == 'admm':
			M, W, obj = self._admm(X, Minit, Winit)

		self.dictionary = M
		self.objective = obj

		return W



	def fit(self, X):

		"""Learns a covariance dictionary from the covariance data X."""

		self.fit_transform(X)
		return self



	def transform(self, X):
		pass


