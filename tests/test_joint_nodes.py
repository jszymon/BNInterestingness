from tempfile import NamedTemporaryFile

import numpy as np

from BNInter.Utils import SparseDistr

from BNInter.BayesNet import BayesNet
from BNInter.BayesNet import BayesSampler
from BNInter.BayesNet import distr_2_str

from BNInter.BayesNet import read_Hugin_file, write_Hugin_file

from BNInter.DataAccess import Attr

import pytest


@pytest.fixture
def basic_bayes_net():
    """Provides a basic BayesNet instance for testing."""
    bn = BayesNet("testNet", [Attr('A', "CATEG", [0,1,2]),
                              Attr('B', "CATEG", [0,1]),
                              Attr('Y', "CATEG", [0,1])])
    bn.addEdge('B', 'Y')
    bn.addEdge('A', 'Y')
    bn["Y"].distr = np.array([
        [[0.1, 0.9], [0.2, 0.8]], #A=0
        [[0.75, 0.25], [0.5, 0.5]], #A=1
        [[0.55, 0.45], [0.6, 0.4]], #A=2
    ])
    bn["A"].distr = np.array([0.5, 0.25, 0.25])
    bn.validate()
    return bn

def test_joint_bn_1(basic_bayes_net):
    bn = basic_bayes_net
    assert not np.allclose(bn.jointP().sum(axis=-1), 1/6)
    bn.addJointDistr(["A", "B"])
    assert "JointNode: ['A', 'B']" in str(bn)
    assert bn.P([0,0,0]) + bn.P([0,0,1]) == pytest.approx(1/6)
    assert np.allclose(bn.jointP().sum(axis=-1), 1/6)

def test_file_output(basic_bayes_net, tmp_path):
    bn = basic_bayes_net
    d = SparseDistr((3,2), {(2,0):0.4, (1,1):0.6}, prior_factor=0.1)
    bn.addJointDistr(["A", "B"], d)
    p = tmp_path / "tmp_bn.net"
    with open(p, "w+") as f:
        write_Hugin_file(bn, f)
    bn2 = read_Hugin_file(str(p))

    assert len(bn.attrs) == len(bn2.attrs)
    assert np.allclose(bn.jointP(), bn2.jointP())

def test_joint_bn_del(basic_bayes_net):
    bn = basic_bayes_net
    orig_distr = bn.jointP()
    bn.addJointDistr(["A", "B"])
    joint_nodes = bn.delJointDistr("A")
    assert len(bn.joint_distrs) == 0
    assert np.allclose(bn.jointP(), orig_distr)
    assert joint_nodes == bn.names_to_numbers(["A", "B"])
