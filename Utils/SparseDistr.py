"""A multivariate sparse joint distribution class.

Sparse representation with uniform prior smooting is used.

"""

import math

import numpy as np

class SparseDistr():
    def __init__(self, shape, distr=None, prior_factor=0):
        """The distributions is a mixture of sparse distribution given
        in distr and a uniform prior.  distr is either a dictionary or
        list of [values,p] lists.  If distr is empty the distribution
        is uniform regardless of the prior_factor.

        """
        if distr is None:
            distr = []
        self.shape = shape
        self.nd = len(self.shape)
        self.size = math.prod(shape)
        self.prior_factor = prior_factor
        self.set_distr(distr)
    def update_distr(self, distr):
        if isinstance(distr, dict):
            distr = list(distr.items())
        old_distr = list((x, p) for x, p in zip(self.X, self.p))
        new_distr = old_distr + distr
        new_distr = list(dict(distr).items()) # make x values unique
        new_distr.sort(key = lambda x: x[0]) # lex sort
        self.X = np.array(list(x[0] for x in new_distr), dtype=int)
        self.p = np.array(list(x[1] for x in new_distr), dtype=float)
        self.normalize()
    def set_distr(self, distr):
        self.X = np.empty((0, self.nd), dtype=int)
        self.p = np.empty((0,), dtype=float)
        self.update_distr(distr)
    def normalize(self):
        """Normalize the distribution."""
        if len(self.p) == 0:
            return
        self.p /= self.p.sum()
    def to_array(self):
        prior = np.full(self.shape, 1/self.size)
        if self.X.shape[0] == 0:
            return prior
        prior *= self.prior_factor
        prior[*[self.X[:,i] for i in range(self.nd)]] += (1-self.prior_factor) * self.p
        return prior
    def sample(self, n):
        rng = np.random.default_rng()
        if self.X.shape[0] == 0:
            from_prior = np.ones(n, dtype=bool)
        else:
            from_prior = (rng.random(n) < self.prior_factor)
        n_from_prior = from_prior.sum()
        s = np.empty((n, self.nd), dtype=int)
        # sample from prior
        for i in range(self.nd):
            s[from_prior,i] = rng.choice(self.shape[i], n_from_prior)
        # sample from distr
        if n_from_prior < n:
            s[~from_prior] = self.X[rng.choice(self.X.shape[0], n-n_from_prior, p=self.p)]
        return s
