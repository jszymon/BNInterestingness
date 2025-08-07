import numpy as np
import random

from ..DataAccess import RecordReader

from .BayesNetGraph import topSort


def randUnivDiscr(distr):
    """Generate a discrete random value from a given distrtibution."""
    x = random.random()
    i = 0
    pdf = distr[0]
    while x > pdf and i + 1 < len(distr):
        i = i + 1
        pdf += distr[i]
    return i

class BayesSampler(RecordReader):
    def __init__(self, bn):
        super(BayesSampler, self).__init__(bn)
        self.bn = bn
        self.toponodes = topSort(bn)
        #self.topomap = [bn.attrnames.index(n) for n in self.toponodes] # map topsort order back to original order
        self.values = [None] * len(bn)

    def next(self):
        for i in self.toponodes:
            node = self.bn[i]
            if len(node.parents) == 0:
                distr = node.distr
            else:
                x = [self.values[p] for p in node.parents]
                distr = node.distr[tuple(x)]
            self.values[i] = randUnivDiscr(distr)
        return list(self.values)
        
    def draw_n_samples(self, n):
        rng = np.random.default_rng()
        sampled_vars = [None] * len(self.bn)
        for i in self.toponodes:
            node = self.bn[i]
            if len(node.parents) == 0:
                distr = node.distr
                x = rng.choice(len(distr), n, p=node.distr)
            else:
                # cumulative distr, skip the last value: it will
                # always be assigned max cetegory
                cdistr = node.distr.cumsum(axis=-1)[...,:-1]
                distr = cdistr[*[sampled_vars[p] for p in node.parents]]
                u = rng.random(n)
                x = distr.shape[1] - (u[:,None] <= distr).sum(axis=1)

                # slow attempt 1
                #x1 = np.array(list(map(lambda a, y: np.searchsorted(a, y, side="right"), distr, u)))
                # slow attempt 2
                #distr = np.column_stack([u, distr])
                #x = np.apply_along_axis(lambda a: np.searchsorted(a[1:], a[0], side="right"), 1, distr)
            sampled_vars[i] = x
        sampled_vars = np.column_stack(sampled_vars)
        return sampled_vars
