# covariance-dictionary
`covdict` is a module for learning a dictionary of covariance matrices from covariance matrix (CM) data. The typical application is to compute a sequence of sample CMs from time-series data, then learn a small dictionary of "fundmental" CMs such that the evolution of the sample CMs can be represented sparsely.

## Optimization problem

We'd like to solve

<p align="center">
  <img src="https://github.com/clarafj/covariance-dictionary/blob/master/equations/opt_prob.png" width="250">
</p>

where X is the input, D is the dictionary, W are the weights, and P and R+ are the feasible sets (details below).

#### Variables

Though the input is an (n_samples, n, n) array of CMs, the actual optimization grunt-work is done in terms of a reformatted, more compact (n_pairs, n_samples) array X where the i-th column is the vectorized upper triangle of the i-th CM in X, and n_pairs = n \* (n + 1) / 2 is the number of upper-triangular entries. 

Borrowing terminology from signal processing, we'll learn an (n_pairs, k) __dictionary__ D of k CM __elements__, as well as a (k, n_samples) array of __weights__ W where the i-th column gives the weights to (approximately) reconstruct the i-th CM in X from the dictionary elements. In practice W tends to end up sparse, even though we don't explicitly enforce sparsity.

#### Objective

The basic idea is to find a dictionary D and weights W that minimize the sum-of-squared-errors ||X - D \* W||^2, where ||.|| gives the Frobenius norm, but there are two constraints on D and W we'd like to have.

#### Constraints

1. Since we want each dictionary element to be a valid CM, or positive semi-definite (PSD), let's call P the set of all valid dictionaries (or to put it inelegantly, the set of all (n_pairs, k) matrices such that each column corresponds to the upper triangle of a PSD matrix). We want D to be in P.
2. For our weights to be easily interpretable, we also want them to be non-negative, or W in R+ where R+ denotes the non-negative orthant for (k, n_samples) matrices.

## Algorithms

We support two algorithms for solving the problem, described below. In practice ADMM appears to converge more consistently across different problem types and sizes.

#### Alternating Directions Method of Multipliers (ADMM)

#### Douglas-Rachford (DR)

#### Alternating Least-Squares (ALS)





