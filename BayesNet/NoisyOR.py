"""Implements a noisy OR gate"""

import numpy

import BayesNet

class NoisyOR(object):
    def __init__(self, names, shape):
        self.names = names
        self.shape = shape
        self.p_true_block = 1
        self.probs = {}
        
    def add_variable(self, varname, block_prob):
        if varname == "true":
            self.p_true_block = block_prob
        self.probs[varname] = block_prob
        
    def get_table(self):
        problist = []
        for i, n in enumerate(self.names):
            if n in self.probs:
                problist.append((i, self.probs[n]))
        tbl = numpy.zeros(self.shape + [2])
        ind = numpy.ndindex(tuple(self.shape))
        for x in ind:
            y = self.p_true_block
            for i, p in problist:
                if x[i] != self.shape[i] - 1:  ### !!! last value is NO !!!
                    y = y * p
            tbl[x+(0,)] = 1 - y
            tbl[x+(1,)] = y
        return tbl



if __name__ == '__main__':
    o = NoisyOR(["C3", "C2"], [2, 2])
    o.add_variable("true", 0.25)
    o.add_variable("C2", 0.1)
    print(BayesNet.distr_2_str(o.get_table(), cond = True))
