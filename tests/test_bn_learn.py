import numpy as np
import pytest

from BNInter.BayesNet import BayesNet
from BNInter.BayesNet.BayesNetLearn import learnProbabilitiesFromData
from BNInter.BayesNet.BayesNetLearn import BDeu
from BNInter.BayesNet.BayesNetLearn import makeIndependentStructure
from BNInter.DataAccess import Attr

@pytest.fixture
def basic_bayes_net():
    """Provides a basic BayesNet instance for testing."""
    bn = BayesNet("testNet",
                  [Attr('A', "CATEG", [0, 1]),
                   Attr('B', "CATEG", [0, 1]),
                   Attr('Y', "CATEG", [0, 1, 2])])
    bn['Y'].set_parents_distr(['B'], np.array([[0.7,0.1,0.2],[0.5,0.3,0.2]]))
    bn.validate()
    return bn

def test_basic_bn_learn(basic_bayes_net):
    bn = basic_bayes_net

    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]], priorN = 0)
    joint_p_array = bn.jointP()
    pa = joint_p_array.sum(axis=(1,2))
    assert np.allclose(pa, [2/3, 1/3])
    pb = joint_p_array.sum(axis=(0,2))
    assert np.allclose(pb, [1/3, 2/3])
    py = joint_p_array.sum(axis=(0,1))
    assert np.allclose(py, [1/3, 1/3, 1/3])

    lnP = BDeu(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    assert lnP <= 0
    makeIndependentStructure(bn)
    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]])
    lnP2 = BDeu(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    assert lnP2 <= 0

def test_bn_w_joint_learn(basic_bayes_net):
    bn = basic_bayes_net
    bn.addJointDistr(("A", "B"))

    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]], priorN = 0)
    joint_p_array = bn.jointP()
    pa = joint_p_array.sum(axis=(1,2))
    assert np.allclose(pa, [2/3, 1/3])
    pb = joint_p_array.sum(axis=(0,2))
    assert np.allclose(pb, [1/3, 2/3])
    py = joint_p_array.sum(axis=(0,1))
    assert np.allclose(py, [1/3, 1/3, 1/3])

    X = [(0,0,1),(0,1,0),(1,1,2)]
    correct_p = np.ones((2,2))
    for x in X:
        correct_p[x[:2]] += 1
    correct_p /= (len(X) + 4)
    learnProbabilitiesFromData(bn, X, priorN = 1)
    joint_p_array = bn.jointP()
    pa = joint_p_array.sum(axis=(1,2))
    assert np.allclose(pa, correct_p.sum(axis=1))
    pb = joint_p_array.sum(axis=(0,2))
    assert np.allclose(pb, correct_p.sum(axis=0))

    lnP = BDeu(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    assert lnP <= 0
    makeIndependentStructure(bn)
    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]], priorN=1)
    lnP2 = BDeu(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    assert lnP2 <= 0

def test_BDeu_joint_score_equivalence(basic_bayes_net):
    """BDeu is score equivalent: a joint node over A, B must score
    the same as the complete DAG A -> B."""
    bn = basic_bayes_net
    data = [[0,0,1],[0,1,0],[1,1,2],[1,1,1],[0,0,0],[1,0,2]]
    BDeu_indep = BDeu(bn, data, priorN=1)
    bn.addEdge("A", "B")
    BDeu_edge = BDeu(bn, data, priorN=1)
    bn.delEdge("A", "B")
    bn.addJointDistr(("A", "B"))
    BDeu_joint = BDeu(bn, data, priorN=1)
    assert BDeu_joint == pytest.approx(BDeu_edge)
    assert BDeu_joint != pytest.approx(BDeu_indep)
