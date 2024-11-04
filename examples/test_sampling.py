import numpy as np

from BayesNet import BayesNet
from BayesNet import BayesSampler
from BayesNet import distr_2_str

from DataAccess import Attr


if __name__ == "__main__":
    print("Sampling from a BN")
    bn = BayesNet("testNet", [Attr('A', "CATEG", [0,1]),
                              Attr('B', "CATEG", [0,1]),
                              Attr('Y', "CATEG", [0,1,2])])
    print(bn)
    bn['Y'].set_parents_distr(['B'], np.array([[0.7,0.1,0.2],[0.5,0.3,0.2]]))
    print(bn)
    print(bn['A'])
    bn.validate()
    print(bn.P([0,0,0]))
    print(distr_2_str(bn.jointP()))

    bs = BayesSampler(bn)
    c = [0, 0, 0]
    for i in range(1_000):
        sample = bs.next()
        c[sample[2]] += 1
        #print sample
    print(c)

    X = bs.draw_n_samples(1_000_000)[:,2]
    print(np.bincount(X))
