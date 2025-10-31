"""Test graph functions of BayesNet"""

import pytest

from BNInter.BayesNet import BayesNet
from BNInter.DataAccess import Attr

from BNInter.BayesNet import ancestors, topSort

@pytest.fixture
def bayesnet_instance():
    """
    Fixture to create a BayesNet instance for testing.
    """
    bn = BayesNet("testGraph",
                  [Attr('a', "CATEG", [0,1]),
                   Attr('b', "CATEG", [0,1]),
                   Attr('c', "CATEG", [0,1]),
                   Attr('d', "CATEG", [0,1])])
    bn.addEdge('a', 'c')
    bn.addEdge('a', 'b')
    bn.addEdge('b', 'c')
    bn.addEdge('d', 'c')
    return bn


def test_ancestors(bayesnet_instance):
    assert ancestors(bayesnet_instance, ['d']) == set()
    assert ancestors(bayesnet_instance, ['c']) == {0, 1, 3}
    assert ancestors(bayesnet_instance, ['b']) == {0}
    assert ancestors(bayesnet_instance, ['a']) == set()


def test_topSort(bayesnet_instance):
    ts = topSort(bayesnet_instance)
    assert (ts == [3,0,1,2]) or (ts == [0,3,1,2])
