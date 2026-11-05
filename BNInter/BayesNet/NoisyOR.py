"""Implements a noisy OR gate.

The last value of each variable (parents and the binary child) means
ON, all other values mean OFF.  Every parent which is ON independently
fails to turn the child ON with its inhibition probability.  The
special variable "true" is an always-ON cause whose inhibition
probability is the probability that the child is OFF when all parents
are OFF (one minus the leak probability)."""

import numpy

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
            # y = probability that all active causes are inhibited
            y = self.p_true_block
            for i, p in problist:
                if x[i] == self.shape[i] - 1:  # last value means ON
                    y = y * p
            tbl[x+(0,)] = y      # OFF
            tbl[x+(1,)] = 1 - y  # ON (last value)
        return tbl



if __name__ == '__main__':
    o = NoisyOR(["C3", "C2"], [2, 2])
    o.add_variable("true", 0.25)
    o.add_variable("C2", 0.1)
    o.add_variable("C3", 0)
    from BNutils import distr_2_str
    print(distr_2_str(o.get_table(), cond = True))
