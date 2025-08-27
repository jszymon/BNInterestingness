import numpy as np
import pytest

from BNInter.BayesNet import BayesNet
from BNInter.BayesNet.BayesNetLearn import learnProbabilitiesFromData
from BNInter.BayesNet.BayesNetLearn import lnP_dataset_cond_network_structure
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

    lnP = lnP_dataset_cond_network_structure(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    assert lnP <= 0
    makeIndependentStructure(bn)
    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]])
    lnP2 = lnP_dataset_cond_network_structure(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
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

    learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]], priorN = 1)
    joint_p_array = bn.jointP()
    pa = joint_p_array.sum(axis=(1,2))
    assert np.allclose(pa, [(1+1+1+1)/(3+4), (1+1)/(3+4)])
    pb = joint_p_array.sum(axis=(0,2))
    #assert np.allclose(pb, [1/3, 2/3])
    py = joint_p_array.sum(axis=(0,1))
    #assert np.allclose(py, [1/3, 1/3, 1/3])

    #lnP = lnP_dataset_cond_network_structure(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    #assert lnP <= 0
    #makeIndependentStructure(bn)
    #learnProbabilitiesFromData(bn, [[0,0,1],[0,1,0],[1,1,2]])
    #lnP2 = lnP_dataset_cond_network_structure(bn, [[0,0,0],[0,0,1],[1,1,1]], priorN=1)
    #assert lnP2 <= 0
