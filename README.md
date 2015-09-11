# covariance-dictionary
`covdict` is a module for learning a dictionary of covariance matrices, inspired by prior algorithms like [this one] (http://www.cs.technion.ac.il/~elad/publications/journals/2004/32_KSVD_IEEE_TSP.pdf) for dictionary learning for sparse image representation. The covariance dictionary can then be used to sparsely represent a complicated time series in terms of its evolving covariance ("network") structure. Check out [this IPython notebook](https://github.com/clarafj/covariance-dictionary/blob/master/demos/zebra_demo.ipynb) for a demo on neural developmental data.

## Optimization problem

We'd like to solve

<p align="center">
  <img src="https://github.com/clarafj/covariance-dictionary/blob/master/equations/opt_prob.png" width="250">
</p>

where _X_ is the input, _D_ is the dictionary, _W_ are the weights, and _P_ and _R+_ are the feasible sets (details below).

#### Variables

Let _n_ be the number of variables in your data. The input is an _(n_samples, n, n)_ array of covariance matrices; for example, sample covariance matrices computed from different sections of time series data. However, the actual optimization grunt-work is done in terms of a reformatted, more compact _(n_pairs, n_samples)_ array _X_ where the _i_-th column is the vectorized upper triangle of the _i_-th covariance matrix in _X_, and _n_pairs_ = _n_ \* (_n_ + 1) / 2 is the number of upper-triangular entries. 

Borrowing terminology from signal processing, we'll learn an _(n_pairs, k)_ __dictionary__ _D_ of _k_ covariance matrix __elements__, as well as a _(k, n_samples)_ array of __weights__ _W_ where the _i_-th column gives the weights to (approximately) reconstruct the _i_-th covariance matrix in _X_ from the dictionary elements. In practice _W_ tends to end up sparse, even though we don't explicitly enforce sparsity.

#### Objective

The basic idea is to find a dictionary _D_ and weights _W_ that minimize the error in reconstructing _X_, but there are two constraints on _D_ and _W_ we'd like to have.

#### Constraints

1. Since we want each dictionary element to be a valid covariance matrix, or positive semi-definite (PSD), let's call _P_ the set of all valid dictionaries (or to put it inelegantly, the set of all _(n_pairs, k)_ matrices such that each column corresponds to the upper triangle of a PSD matrix). We want _D_ to be in _P_. (For applications where analyzing the correlation matrix makes more sense, the correlation matrix constraint is also supported for ADMM.)
2. For our weights to be easily interpretable, we also want them to be non-negative, or _W_ in _R+_ where _R+_ denotes the non-negative orthant for _(k, n_samples)_ matrices.

## Algorithms

We support two algorithms for solving the problem, described below. In practice ADMM appears to converge more consistently across different problem types and sizes.

#### Alternating Directions Method of Multipliers (ADMM)

#### Douglas-Rachford (DR)

(Coming soon. Very similar in formulation to ADMM, but for some problems in signal processing, like [phase retrieval](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-19-7-1334) the literature suggests it works better.)

#### Alternating Least-Squares (ALS)

