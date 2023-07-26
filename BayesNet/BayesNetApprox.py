import numpy
import random

from DataAccess import Attr, RecordReader

from BayesNet import BayesNode, BayesNet
from .BNutils import distr_2_str
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
        


if __name__ == "__main__":
    print("Sampling from a BN")
    bn = BayesNet("testNet", [Attr('A', "CATEG", [0,1]),
                              Attr('B', "CATEG", [0,1]),
                              Attr('Y', "CATEG", [0,1,2])])
    print(bn)
    bn['Y'].set_parents_distr(['B'], numpy.array([[0.7,0.1,0.2],[0.5,0.3,0.2]]))
    print(bn)
    print(bn['A'])
    bn.validate()
    print(bn.P([0,0,0]))
    print(distr_2_str(bn.jointP()))

    c = [0, 0, 0]
    for i in range(1000):
        r = randUnivDiscr([0.1, 0.6, 0.3])
        c[r] += 1
    print(c)

    bs = BayesSampler(bn)
    c = [0, 0, 0]
    for i in range(10000):
        sample = bs.next()
        c[sample[2]] += 1
        #print sample
    print(c)
