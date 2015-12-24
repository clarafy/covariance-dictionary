""" 
Class and functions for learning dictionary of covariances matrices
"""

import warnings
import sys
import time
from numpy import (arange, cov, diag, diag_indices, dot, dsplit, dstack, empty, finfo, fmax, 
    hstack, identity, logical_or, max, mod, sqrt, where, triu_indices, vstack, zeros)
from numpy.random import rand
from scipy.linalg import eigh, inv, norm



def plot_element(element, coords, thresh=0.5, size=200, colors=None):

    # TODO: If covariance and not correlation element, need to normalize
    # line widths (commented out below).

    from numpy import array, percentile
    import matplotlib.pyplot as plt

    n = element.shape[0]
    if colors is None:
        colors = 'DarkGray'

    edges_pos = where(element > percentile(element[element > 0], 100 * thresh))
    # alpha_pos = element[edges_pos] / max(element[element > 0])
    width_pos = element[edges_pos] * 5
    edges_neg = where(-element > percentile(-element[element < 0], 100 * thresh))
    # alpha_neg = -element[edges_neg] / max(-element[element < 0])
    width_neg = -element[edges_neg] * 5

    for i, edge in enumerate(array(edges_pos).T):
        plt.plot(coords[edge, 0], coords[edge, 1], 'YellowGreen', linewidth=width_pos[i]);

    for i, edge in enumerate(array(edges_neg).T):
        plt.plot(coords[edge, 0], coords[edge, 1], 'PaleVioletRed', linewidth=width_neg[i]);

    plt.scatter(coords[:, 0], coords[:, 1], s=size, linewidth=0, c=colors);
    plt.axis('off')



def pack(vec, n=None, half=False):

    # Reforms a vector into the upper triangle
    # of a symmetric matrix.

    if n is None:
        n = npair2n(vec.size)

    packed = zeros((n, n))
    packed[triu_indices(n)] = vec

    if not half:
        symm = packed + packed.T
        symm[diag_indices(n)] = packed[diag_indices(n)]
        packed = symm

    return packed



def pack_samples(unpacked, n=None, half=False):

    # Packs each column into the upper triangle of
    # a symmetric and returns 3D stack.

    n_pair, n_samp = unpacked.shape
    if n is None:
        n = npair2n(n_pair)
    return dstack([pack(col, n=n, half=half) for col in unpacked.T])



def unpack_samples(packed):

    # Unpacks the upper triangle of each n x n matrix in a 3D stack
    # into a column of the 2D output.

    n, _, n_samp = packed.shape
    return vstack([mat[triu_indices(n)] for mat in packed.T]).T



def proj_psd(A):

    # Projects a symmetric matrix to nearest
    # positive semi-definite matrix.

    d, U = eigh(A, lower=False)
    U = U[:, d > 0]
    d = d[d > 0]
    Aproj = dot(dot(U, diag(d)), U.T)

    return Aproj

def proj_corr(A, max_iter=100, tol=1e-6):

    # Projects a symmetric matrix to the nearest correlation
    # matrix (PSD matrix with equality constraints on
    # the diagonal and bound constraints on the off-diagonal)
    # using Dykstra's algorithm, as described in Higham (2002)
    # "Computing the nearest correlation matrix: A problem from finance".

    # How exactly is Dykstra's different from ADMM for two projections?

    # Is there a norm under which the closest correlation matrix to a
    # covariance matrix is its own correlation matrix? Should do projection
    # under this norm.

    n = A.shape[0]
    deltaS = zeros((n, n))
    Y = A
    X = zeros((n, n)) 

    triu_idx = triu_indices(n)
    diag_idx = diag_indices(n) 

    for n_iter in range(max_iter):

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

        if max([diffX, diffY, diffXY]) < tol:
            break

    if n_iter == max_iter - 1:
            warnings.warn("Max iterations reached in correlation matrix projection.")

    return Y



def proj_col_psd(A, correlation=False):

    # Projects every column of a matrix to the upper-triangle
    # of the nearest positive semi-definite or correlation matrix.

    n_pair, n_col = A.shape
    n = npair2n(n_pair)

    Aproj = zeros((n_pair, n_col))
    mat_triu = zeros((n, n))
    triu_idx = triu_indices(n)

    if correlation:
        for col_idx in range(n_col):

            # Reconstruct symmetric matrix (only need half).
            mat_triu[triu_idx] = A[:, col_idx]
            Aproj[:, col_idx] = proj_corr(mat_triu)[triu_idx]
    else:
        for col_idx in range(n_col):

            # Reconstruct symmetric matrix (only need half).
            mat_triu[triu_idx] = A[:, col_idx]
            Aproj[:, col_idx] = proj_psd(mat_triu)[triu_idx]

    return Aproj



def npair2n(n_pair):

    # Number of nodes from number of upper triangular entries.

    return int((sqrt(1 + 8 * n_pair) - 1) / 2)



class CovarianceDictionary(object):

    """
    Learns a dictionary of covariance matrices

    Parameters
    ----------
    k : int, optional, default = 2
        Number of dictionary elements

    method : str in {'admm', 'als', 'pgm'}, optional, default = 'admm'
        Specifies which optimization algorithm to use. Alternating least-squares
        ('als'), alternating directions method of multipliers ('admm'), and
        projected gradient method ('pgm') supported
 
    init : str in {'kmeans', 'rand'}, optional, default = 'kmeans'
        Specifies how to initialize the dictionary and weights. 'k-means'
        clusters the input data to initialize as in Wild, Curry, & Dougherty (2004),
        and 'rand' initializes to random linear combinations of input.
        Random initialization completely derails ADMM, takes forever to converge.

    max_iter : int, optional, default = None
        Maximum number of iterations. If None, 200 for ALS and 6000 for ADMM

    tol : float, optional, default = 1e-5 for ALS and ADMM, 1e-3 for PGM
        Stopping tolerance on projected gradient norm for ALS and PGM, and objective for ADMM.
        Taking it easy on PGM, poor thing tends to converge at much greater objective values,
        consistent with Lin (2007).

    nls_max_iter : int, optional, default = 2000
        Maximum number of iterations for the non-negative least-squares subproblem
        in ALS

    psdls_max_iter : int, optional, default = 2000
        Maximum number of iterations for the positive semidefinite least-squares
        subproblem in ALS

    nls_beta : float > 0 and < 1, optional, default = 0.2
        Step size search parameter for the non-negative least-squares subproblem
        in ALS, as in "Armijo rule along the projection arc" in Bertsekas (1999)
        Larger values mean larger jumps in searching for step size, so
        can speed up convergence but may be less accurate

    psdls_beta : float > 0 and < 1, optional, default = 0.2
        Step size search parameter for the positive-semidefinite least-squares subproblem
        in ALS, as in "Armijo rule along the projection arc" in Bertsekas (1999).
        Larger values mean larger jumps in searching for step size, so
        can speed up convergence but may be less accurate. Empirically
        larger psdls_beta affects accuracy more so than nls_beta

    pgm_beta : float > 0 and < 2, optional, default = 0.5
    	Step size search parameter for the projected gradient method, as in
    	"Armijo rule along the projection arc" in Bertsekas (1999).

    correlation : boolean, optional, default = False
        Whether to find dictionary of correlation matrices rather
        than covariance matrices. Supported for both ALS and ADMM,
        but takes long as chickens for ALS so only use ADMM

    admm_gamma : float, optional, default = 1
        Constant on step size rule for ADMM 

    admm_alpha : float, optional, default = 47.75
        Scaling constant on penalty on proximal term ||U - D||_F^2 for ADMM

    verbose : boolean, optional, default = False
        Whether to print algorithm progress (projected gradient norm for
        ALS, objective for ADMM)

    time : boolean, optional, default = False 
        Whether to time each iteration

    obj_tol : float, optional, default = None
        Stopping condition on raw objective value. If None, stopping rule is 
        instead based on objective decrease for ADMM and projected gradient norm for ALS.
        Should only be used when true minimum objective value is known

    Attributes
    ----------
    dictionary: array, shape (n, n, k)
        Dictionary of covariance or correlation matrices where each column 
        gives the upper triangle of a dictionary element

    objective: array, shape (n_iter)
        Value of objective ||X - DW||_F / ||X||_F at each iteration

    References
    ----------
    Bertsekas. 1976.
    Bertsekas. 1999.
    Calamai & More (1999)
    Lin, C.-J. & More (1999)
    Lin, C.-J. (2007). "Projected gradient methods for non-negative matrix factorization".
    	Neural Computation.
    Wild, Curry, & Dougherty. (2004). "Improving non-negative matrix factorizations through
        structured initialization"
    Xu, Y., Yin, W., Wen, Z., & Zhang, Y. (2012). "An alternating direction algorithm for matrix 
        completion with nonnegative factors". 

    """

    def __init__(self, k=2, method='admm', init='kmeans', max_iter=None, tol=None, 
        verbose=False, obj_tol=None, time=False, 
        nls_beta=0.2, psdls_beta=0.2, nls_max_iter=2000, psdls_max_iter=2000,
        pgm_beta=0.5, correlation=False, admm_gamma=1, admm_alpha=47.75):

        if method not in ('als', 'admm', 'pgm'):
            raise ValueError(
                                'Invalid method: got %r instead of one of %r' %
                                (method, ('als', 'admm', 'pgm')))
        self.method = method

        if init not in ('kmeans', 'rand'):
            raise ValueError(
                                'Invalid initialization: got %r instead of one of %r' %
                                (init, ('kmeans', 'rand')))
        self.init = init

        if max_iter is None:
            if self.method == 'als':
                self.max_iter = 200
            elif self.method == 'admm':
                self.max_iter = 6000
            elif self.method == 'pgm':
            	self.max_iter = 600
        else:
            self.max_iter = max_iter

        if tol is None:
        	self.tol = 1e-3 if self.method == 'pgm' else 1e-5
        else:
        	self.tol = tol

        self.k = k
        self.nls_max_iter = nls_max_iter
        self.psdls_max_iter = psdls_max_iter
        self.nls_beta = nls_beta
        self.psdls_beta = psdls_beta
        self.pgm_beta = pgm_beta
        self.correlation = correlation
        self.admm_gamma = admm_gamma
        self.admm_alpha = admm_alpha
        self.verbose = verbose
        self.obj_tol = obj_tol
        self.time = time
        self.dictionary = None
        self.objective = None



    def _initialize(self, X):

        # Initializes the dictionary D and weights W randomly or using k-means,
        # as in Wild, Curry, & Dougherty (2004) "Improving non-negative 
        # matrix factorizations through structured initialization".

        from sklearn.cluster import KMeans

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)

        if self.init == 'kmeans':

        	# Initialize modules to k-means cluster centroids.

            Xnorm = X / hstack([norm(col) for col in X.T]) 
            km = KMeans(n_clusters=self.k).fit(Xnorm.T)
            centroids = km.cluster_centers_.T
            Dinit = proj_col_psd(centroids, self.correlation)

        elif self.init == 'rand':

            # Initialize modules to random linear combinations
            # of input covariances.
            Dinit = dot(X, rand(n_samp, self.k))

        Winit, _, _ = self._nls_subproblem(X, Dinit, rand(self.k, n_samp), 1e-3)
        return Dinit, Winit



    def _admm(self, X, Dinit, Winit):

        # Solves for covariance module and weights using ADMM. Reformulate
        # original optimization problem

        # minimize ||X - DW||_F
        # subject to
        # each column of D is a PSD matrix
        # each element of W is non-negative

        # as

        # minimize ||X - DW||_F
        # subject to
        # D = U
        # W = V
        # each column of U is a PSD matrix
        # each element of V is non-negative

        # and sequentially minimize the augmented Lagrangian
        # w.r.t U, V, D, and W.

        # Can also solve problem under constraint of correlation
        # matrices rather than general PSD matrices.

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)
        max_dim = max([n_pair, n_samp])

        D = Dinit
        W = Winit
        V = Winit
        Lambda = zeros((n_pair, self.k))
        Pi = zeros((self.k, n_samp))

        normX = norm(X)
        objective = empty(self.max_iter)
        obj_prev = finfo('d').max

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        for n_iter in range(self.max_iter):

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping condition.
            if self.obj_tol is None:
                if (abs(obj - obj_prev) / fmax(1, obj_prev) < self.tol or
                        obj < self.tol or
                        obj > obj_prev):
                    break
            elif obj < self.obj_tol:
                break

            obj_prev = obj

            if self.verbose:
                if mod(n_iter, 100) == 0:
                    print 'Iter: %i. Objective: %f.' % (n_iter, obj)
                    sys.stdout.flush()

            # Step size rule
            # alpha = self.admm_alpha * normX * max_dim / (n_iter + 1)
            # beta = alpha * n_samp / n_pair
            alpha = self.admm_alpha * max_dim / (n_iter + 1)
            beta = alpha * n_samp / n_pair

            # Primal variable updates
            U = dot(dot(X, V.T) + alpha * D - Lambda, inv(dot(V, V.T) + alpha * identity(self.k)))
            V = dot(inv(dot(U.T, U) + beta * identity(self.k)), dot(U.T, X) + beta * W - Pi)

            D = proj_col_psd(U + Lambda / alpha, self.correlation)
            W = fmax(V + Pi / beta, 0)

            # Dual variable updates
            Lambda = Lambda + self.admm_gamma * alpha * (U - D)
            Pi = Pi + self.admm_gamma * beta * (V - W)


        if self.verbose:
            print 'Iter: %i. Objective: %f.' % (n_iter, obj)
            sys.stdout.flush()

        objective = objective[: n_iter + 1]
        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times



    def _nls_subproblem(self, X, D, Winit, tol):

        # Update weights by solving non-negative least-squares
        # using projected gradient descent (basically a transposed 
        # version of scikit-learn's NMF _nls_subproblem method).

        W = Winit
        DtX = dot(D.T, X)
        DtD = dot(D.T, D)

        pg_norm = empty(self.nls_max_iter)
        # in_iter = empty(self.max_iter)

        alpha = 1

        for n_iter in range(self.nls_max_iter):

            grad = dot(DtD, W) - DtX

            # Stopping condition on projected gradient norm.
            # Multiplication with a boolean array is more than twice
            # as fast as indexing into grad.
            pgn = norm(grad * logical_or(grad < 0, W > 0))
            pg_norm[n_iter] = pgn

            if pgn < tol:
                break

            Wold = W

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
            # condition in Lin (2007) Eq. (16)).
            for inner_iter in range(10):

                # Gradient step.
                Wnew = W - alpha * grad

                # Projection step.
                Wnew *= Wnew > 0

                # Check Lin (2007) Eq. (16) condition.
                d = Wnew - W
                gradd = dot(grad.ravel(), d.ravel())
                dQd = dot(dot(DtD, d).ravel(), d.ravel())
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0

                # 1.1 If initially not sufficient decrease, then...
                # 2.1 If initially sufficient decrease, then...
                if inner_iter == 0:
                    decr_alpha = not suff_decr

                if decr_alpha:
                    # 1.3 ...there is sufficient decrease.
                    if suff_decr:
                        W = Wnew
                        break
                    # 1.2 ...decrease alpha until...
                    else:
                        alpha *= self.nls_beta

                # 2.3 ...there is not sufficient decrease.
                elif not suff_decr or (Wold == Wnew).all():
                    W = Wold
                    break

                # 2.2 ...increase alpha until...
                else:
                    alpha /= self.nls_beta
                    Wold = Wnew

            # in_iter[n_iter] = inner_iter

        if n_iter == self.nls_max_iter - 1:
            warnings.warn("Max iterations reached in NLS subproblem.")

        pg_norm = pg_norm[: n_iter + 1]
        # in_iter = in_iter[: n_iter]

        return W, grad, n_iter 
        

    def _psdls_subproblem(self, X, Dinit, W, tol):

        # Update modules by solving column-wise positive-semidefinite (PSD)
        # constrained least-squares using projected gradient descent:

        # minimize ||X - D * W||_F
        # subject to the constraint that every column of D
        # corresponds to the upper triangle of a PSD matrix.

        n_pair, n_samp = X.shape
        n = npair2n(n_pair)

        D = Dinit
        WWt = dot(W, W.T)
        XWt = dot(X, W.T)
        pg_norm = empty(self.psdls_max_iter)
        # in_iter = empty(self.psdls_max_iter)
        
        alpha = 1

        for n_iter in range(self.psdls_max_iter):

            gradD = dot(D, WWt) - XWt

            # Stopping condition on projected gradient norm.
            pgn = norm(proj_col_psd(D - gradD, self.correlation) - D)
            pg_norm[n_iter] = pgn

            if pgn < tol:
                break

            Dold = D

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999), using shortcut
            # condition in Lin (2007) Eq. (16).)
            for inner_iter in range(20):

                # Gradient and projection steps.
                Dnew = proj_col_psd(D - alpha * gradD, self.correlation)

                d = Dnew - D
                gradd = dot(gradD.ravel(), d.ravel())
                dQd = dot(dot(d, WWt).ravel(), d.ravel())
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0

                # 1.1 If initially not sufficient decrease, then...
                # 2.1 If initially sufficient decrease, then...
                if inner_iter == 0:
                    decr_alpha = not suff_decr

                if decr_alpha:
                    # 1.3 ...there is sufficient decrease.
                    if suff_decr:
                        D = Dnew
                        break
                    # 1.2 ...decrease alpha until...
                    else:
                        alpha *= self.psdls_beta

                # 2.3 ...there is not sufficient decrease.
                elif not suff_decr or (Dold == Dnew).all():
                    D = Dold
                    break

                # 2.2 ...increase alpha until...
                else:
                    alpha /= self.psdls_beta
                    Dold = Dnew

            # in_iter[n_iter] = inner_iter

        if n_iter == self.psdls_max_iter - 1:
            warnings.warn("Max iterations reached in PSDLS subproblem.")

        pg_norm = pg_norm[: n_iter + 1]
        # in_iter = in_iter[: n_iter]

        return D, gradD, n_iter 


    def _als(self, X, Dinit, Winit):

        # Solves for covariance module and weights using
        # alternating constrained least-squares. Same framework
        # as scikit-learn's ALS for NMF.

        n_pair, n_mat = X.shape
        n = npair2n(n_pair)
        
        D = Dinit
        W = Winit

        # Initial gradient.
        gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
        gradW = dot(dot(D.T, D), W) - dot(D.T, X)
        init_grad_norm = norm(vstack((gradD, gradW.T)))
        
        if self.verbose:
            print "Initial gradient norm: %f." % init_grad_norm
            sys.stdout.flush()

        # Stopping tolerances for constrained ALS subproblems.
        tolD = self.tol * init_grad_norm
        tolW = tolD

        normX = norm(X)
        objective = empty(self.max_iter)
        # pg_norm = empty(self.max_iter)

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        for n_iter in range(self.max_iter):

            # Stopping criterion, based on Calamai & More (1987) Lemma 3.1(c)
            # (stationary point iff projected gradient norm = 0).
            pgradW = gradW * logical_or(gradW < 0, W > 0)
            pgradD = proj_col_psd(D - gradD, self.correlation) - D
            pgn = norm(vstack((pgradD, pgradW.T)))
            # pg_norm[n_iter] = pgn

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping condition.
            if self.obj_tol is None:
                if pgn < self.tol * init_grad_norm:
                    break
            elif obj < self.obj_tol:
                break

            if self.verbose:
                print 'Iter: %i. Projected gradient norm: %f. Objective: %f.' % (n_iter, pgn, obj)
                sys.stdout.flush()

            # Update modules.
            D, gradD, iterD = self._psdls_subproblem(X, D, W, tolD)
            if iterD == 0:
                tolD = 0.1 * tolD

            # Update weights.
            W, gradW, iterW = self._nls_subproblem(X, D, W, tolW)
            if iterW == 0:
                tolH = 0.1 * tolW

        if self.verbose:
            print 'Iter: %i. Final projected gradient norm %f. Final objective %f.' % (n_iter, pgn, obj)
            sys.stdout.flush()

        objective = objective[: n_iter + 1]
        # pg_norm = pg_norm[: n_iter + 1]

        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times



    def _pgm(self, X, Dinit, Winit):

    	# Solves for covariance module and weights using
        # a projected gradient method.

        n_pair, n_mat = X.shape
        n = npair2n(n_pair)
        
        D = Dinit
        W = Winit

        # Initial gradient.
        gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
        gradW = dot(dot(D.T, D), W) - dot(D.T, X)
        init_grad_norm = norm(vstack((gradD, gradW.T)))
        
        if self.verbose:
            print "Initial gradient norm: %f." % init_grad_norm
            sys.stdout.flush()

        normX = norm(X)
        objective = empty(self.max_iter)

        if self.time:
            times = empty(self.max_iter)
            t = time.time()
        else:
            times = None

        alpha = 1

        for n_iter in range(self.max_iter):

            # Compute projected gradient for stopping condition. 
            pgradW = gradW * logical_or(gradW < 0, W > 0)
            pgradD = proj_col_psd(D - gradD, self.correlation) - D
            pgn = norm(vstack((pgradD, pgradW.T)))

            # Record objective.
            obj = norm(X - dot(D, W)) / normX
            objective[n_iter] = obj

            if self.time:
                times[n_iter] = time.time() - t

            # Stopping criterion, based on Calamai & More (1987) Lemma 3.1(c)
            # (stationary point iff projected gradient norm = 0).
            if self.obj_tol is None:
                if pgn < self.tol * init_grad_norm:
                    break
            elif obj < self.obj_tol:
                break

            if self.verbose:
                print 'Iter: %i. Projected gradient norm: %f. Objective: %f.' % (n_iter, pgn, obj)
                sys.stdout.flush()

            Wold = W
            Dold = D

            # Search for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999)), using
            # "warm-start" trick in Lin & More (1999).
            for inner_iter in range(20):

            	# Proposed updates of dictionary and weights
            	# (gradient step and projection step)
                # Gradient step.
                Wnew = W - alpha * gradW
                Wnew *= Wnew > 0
                Dnew = proj_col_psd(D - alpha * gradD, self.correlation)

                # Check for sufficient decrease.
                obj_old = pow(obj * normX, 2)
                obj_new = pow(norm(X - dot(Dnew, Wnew)), 2)
                W_thresh_decr = dot(gradW.ravel(), (Wnew - W).ravel())
                D_thresh_decr = dot(gradD.ravel(), (Dnew - D).ravel())
                suff_decr = obj_new - obj_old < 0.01 * (W_thresh_decr + D_thresh_decr)

                # 1.1 If initially not sufficient decrease, then...
                # 2.1 If initially sufficient decrease, then...
                if inner_iter == 0:
                    decr_alpha = not suff_decr

                if decr_alpha:
                    # 1.3 ...there is sufficient decrease.
                    if suff_decr:
                        W = Wnew
                        D = Dnew
                        break
                    # 1.2 ...decrease alpha until...
                    else:
                        alpha *= self.pgm_beta

                # 2.3 ...there is not sufficient decrease.
                elif not suff_decr or ((Wold == Wnew).all() and (Dold == Dnew).all()):
                    W = Wold
                    D = Dold
                    break

                # 2.2 ...increase alpha until...
                else:
                    alpha /= self.pgm_beta
                    Wold = Wnew
                    Dold = Dnew

            gradD = dot(D, dot(W, W.T)) - dot(X, W.T)
            gradW = dot(dot(D.T, D), W) - dot(D.T, X)

        if self.verbose:
            print 'Iter: %i. Final projected gradient norm %f. Final objective %f.' % (n_iter, pgn, obj)
            sys.stdout.flush()

        objective = objective[: n_iter + 1]
        if self.time:
            times = times[: n_iter + 1]

        return D, W, objective, times



    def fit_transform(self, X):
            
        """Learns a dictionary from data X and returns dictionary weights


        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        W : array, shape (k, n_samp)
            Weights of dictionary elements for input

        """

        X = unpack_samples(X) # unpack into (n_pair, n_samp) array
        Dinit, Winit = self._initialize(X)

        if self.method == 'als':
            D, W, obj, times = self._als(X, Dinit, Winit)
        elif self.method == 'admm':
            D, W, obj, times = self._admm(X, Dinit, Winit)
        elif self.method == 'pgm':
            D, W, obj, times = self._pgm(X, Dinit, Winit)

        self.dictionary = pack_samples(D)
        self.D = D
        self.objective = obj
        self.times = times


        return W



    def fit(self, X):

        """Learns a covariance dictionary from the covariance data X


        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        self : object
            The instance itself

        """

        self.fit_transform(X)
        return self



    def transform(self, X):

        """
        Computes the dictionary weights for input covariance data

        Parameters
        ----------
        X : array, shape (n, n, n_samp)
            Input covariance data as a stack of n_samp covariance matrices

        Returns
        -------
        W : array, shape (k, n_samp)
            Weights of dictionary elements for each input

        """

        X = unpack_samples(X)
        if self.dictionary is not None:
            W = self._nls_subproblem(X, self.dictionary, rand(k, n_samp), 1e-3)
        else:
            W = self.fit_transform(X)

        return W


