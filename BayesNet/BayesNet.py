#!/usr/bin/env python

import numpy

from DataAccess import Attr, AttrSet
from .BNutils import blockiter, distr_2_str

    



class BayesNode(Attr):
    def __init__(self, bnet, attr):
        """Create Bayesian network node.

        Initially the node has no parents and a uniform distribution."""
        super(BayesNode, self).__init__(attr.name, attr.type, attr.domain)
        self.bnet = bnet # reference to BayesNet the node is in
        self.parents = []
        nd = len(self.domain)
        self.distr = numpy.zeros(nd) + 1.0/nd

    def fix_data(self):
        """Fix node params which are known only after the full network is created"""

    def P(self, x):
        """Get (conditional) probability for given vector x."""
        idx = [x[i] for i in self.parents]
        return self.distr[tuple(idx)]
    def normalizeProbabilities(self):
        """Normalize all conditional probabilities in a node so that
        they add up to 1.0."""
        ri = len(self.domain)
        i = 0
        flat = numpy.ravel(self.distr)
        while i < len(flat):
            s = sum(flat[i:i+ri])
            for j in range(i, i+ri):
                flat[j] /= s
            i += ri

    def set_parents_distr(self, parents, distr):
        if len(parents) > 0 and isinstance(parents[0], str):
            parents = self.bnet.names_to_numbers(parents)
        self.parents = parents
        # TODO: verify distr size
        self.distr = distr

    def __str__(self):
        ret = super(BayesNode, self).__str__()
        ret += "\nParents: " + str([self.bnet[i].name for i in self.parents])
        ret += "\nDistribution:\n" + distr_2_str(self.distr, cond = True)
        return ret



class BayesNet(AttrSet):
    def __init__(self, name = "", attrs = None):
        nodes = [BayesNode(self, a) for a in attrs]
        super(BayesNet, self).__init__(name, nodes)



    def addEdge(self, src_name, dst_name):
        """Add an edge from attribute src_name to attribute dst_name.""" 
        i = self.names_to_numbers([src_name])[0]
        j = self.names_to_numbers([dst_name])[0]
        src_node = self[src_name]
        dst_node = self[dst_name]
        if i in dst_node.parents:
            raise RuntimeError("Nodes already connected")
        # TODO: check if cycles arent introduced
        dst_node.distr = numpy.array([dst_node.distr] * len(src_node.domain))
        dst_node.parents.insert(0,i)

    def delEdge(self, src_name, dst_name):
        """Delete the edge from attribute src_name to attribute dst_name.""" 
        i = self.names_to_numbers([src_name])[0]
        j = self.names_to_numbers([dst_name])[0]
        src_node = self[src_name]
        dst_node = self[dst_name]
        if i not in dst_node.parents:
            raise RuntimeError("Edge does not exist")
        dst_node.distr = numpy.sum(dst_node.distr, dst_node.parents.index(i)) / 2
        del dst_node.parents[dst_node.parents.index(i)]

    


    def validate(self, err = 0.00001):
        """Validates the network.

        Currently only checks that each distribution is a conditional
        distribution."""
        # TODO: acyclicity, distr sizes
        for n in self:
            ri = len(n.domain)
            for subdistr in blockiter(numpy.ravel(n.distr), ri):
                if abs(sum(subdistr) - 1.0) > err:
                    print("error in node", n.name, "prob sum=", sum(subdistr))


    def normalizeProbabilities(self):
        """Normalize all conditional probabilities in all nodes so
        that they add up to 1.0."""
        for n in self:
            n.normalizeProbabilities()

    def P(self, x):
        """Return the probability of input vector x"""
        P = 1.0
        for i, n in enumerate(self):
            P *= n.P(x)[x[i]]
        return P

    def get_shape(self):
        """Get the shape of the joint distribution."""
        shape = [len(a.domain) for a in self]
        return shape

    def jointP(self):
        """Return the numpy for the joint distribution of the network."""
        shape = tuple(self.get_shape())
        d = numpy.zeros(shape)
        for x in numpy.ndindex(shape):
            d[x] = self.P(x)
        return d
    
    def __str__(self):
        ret = ""
        ret += "Bayesian network: " + self.name
        ret += "\nNodes:\n"
        ret += "\n".join([str(node) for node in self])
        return ret

if __name__ == '__main__':
    bn = BayesNet("testNet",
                  [Attr('A', "CATEG", [0,1]),
                   Attr('B', "CATEG", [0,1]),
                   Attr('Y', "CATEG", [0,1,2])])
    print(bn)
    print('------------------')

    bn['Y'].set_parents_distr(['B'], numpy.array([[0.7,0.1,0.2],[0.5,0.3,0.2]]))
    #bn['Y'].set_parents_distr(['B'], numpy.array([[0.7,0.1,0.2],[0.5,0.3,1.2]]))  # wrong sum to test validate()
    print(bn)
    print(bn['A'])
    bn.validate()
    print(bn.P([0,0,0]))
    print(distr_2_str(bn.jointP()))

    bn.addEdge('A', 'B')
    print(bn)
    bn.delEdge('A', 'B')
    print(bn)
