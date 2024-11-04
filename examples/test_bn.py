import numpy as np

from BayesNet import BayesNet
from BayesNet import distr_2_str

from DataAccess import Attr



if __name__ == '__main__':
    bn = BayesNet("testNet",
                  [Attr('A', "CATEG", [0,1]),
                   Attr('B', "CATEG", [0,1]),
                   Attr('Y', "CATEG", [0,1,2])])
    print(bn)
    print('------------------')

    bn['Y'].set_parents_distr(['B'], np.array([[0.7,0.1,0.2],[0.5,0.3,0.2]]))
    #bn['Y'].set_parents_distr(['B'], np.array([[0.7,0.1,0.2],[0.5,0.3,1.2]]))  # wrong sum to test validate()
    print(bn)
    print(bn['A'])
    bn.validate()
    print(bn.P([0,0,0]))
    print(distr_2_str(bn.jointP()))

    bn.addEdge('A', 'B')
    print(bn)
    bn.delEdge('A', 'B')
    print(bn)
    bn['Y'].del_all_parents()
    print(bn)
