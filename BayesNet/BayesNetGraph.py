"""Graph Theory related aspects of Bayesian networks, like topological
sorting, cyclicity checks."""

from DataAccess import Attr
from BayesNet import BayesNet


def ancestors(bn, nodes):
    """Returns a set of ancestors of attributes in attrnames.

    More precisely the union of ancestor sets of each attribute in
    attrnames"""
    if isinstance(nodes[0], str):
        nodes = bn.names_to_numbers(nodes)
    anc = set()
    seed = nodes
    while len(seed) > 0:
        parents = set()
        for i in seed:
            node = bn[i]
            parents |= set(node.parents)
        parents = parents - anc  # skip already considered nodes
        anc |= parents
        seed = parents
    return anc - set(nodes)

def topSort(bn):
    """Returns names of nodes of bn in topological sort order.

    Throws an exception if the network is cyclic."""
    sorted = []
    indegrees = {}
    S = []  # list of nodes with in-degrees ==0
    for i, n in enumerate(bn):
        l = len(n.parents)
        if l == 0:
            S.append(i)
        else:
            indegrees[i] = l
    while len(S) > 0:
        i = S.pop()
        sorted.append(i)
        for i2, indg in list(indegrees.items()):
            if i in bn[i2].parents:
                indg = indg - 1
                if indg == 0:
                    S.append(i2)
                    del indegrees[i2]
                else:
                    indegrees[i2] = indg
    if len(indegrees) > 0:
        raise RuntimeError("Cycle in Bayesian network")
    return sorted

if __name__ == "__main__":
    bn = BayesNet("testGraph",
                  [Attr('a', "CATEG", [0,1]),
                   Attr('b', "CATEG", [0,1]),
                   Attr('c', "CATEG", [0,1]),
                   Attr('d', "CATEG", [0,1])])
    
    bn.addEdge('a', 'c')
    bn.addEdge('a', 'b')
    bn.addEdge('b', 'c')
    bn.addEdge('d', 'c')

    print(ancestors(bn, ['d']))
    print(ancestors(bn, ['c']))
    print(ancestors(bn, ['b']))
    print(ancestors(bn, ['a']))

    #bn.addEdge('d', 'c')  # add a cycle
    print(topSort(bn))
