import numpy as np

from BayesNet import BayesNet
from BayesNet import BayesSampler

from BayesNet import BayesNode, BayesNet
from BayesNet import distr_2_str

from DataAccess import Attr, RecordReader



bn = BayesNet("testNet", [Attr('A', "CATEG", [0,1,2]),
                          Attr('B', "CATEG", [0,1]),
                          Attr('Y', "CATEG", [0,1])])
print(bn['A'])
bn.addEdge('B', 'Y')
bn.addEdge('A', 'Y')
bn["Y"].distr = np.array([
    [[0.1, 0.9], [0.2, 0.8]], #A=0
    [[0.75, 0.25], [0.5, 0.5]], #A=1
    [[0.55, 0.45], [0.6, 0.4]], #A=2
])
print(bn)
bn.validate()
print(bn.P([0,0,0]))
print(distr_2_str(bn.jointP()))

print("---------------------------------")
bn.addJointDistr(["A", "B"])
print(bn)
