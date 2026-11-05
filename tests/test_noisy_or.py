"""Tests for Noisy-OR distributions.

Convention: the last value of each variable means ON, all other values
mean OFF.  Index 0 of the child is OFF, index 1 is ON."""

import numpy as np
import pytest

from BNInter.BayesNet import BayesNet, read_Hugin_file, write_Hugin_file
from BNInter.BayesNet.NoisyOR import NoisyOR
from BNInter.DataAccess import Attr


def test_noisy_or_no_inhibitors():
    """Without any variables the output does not depend on parents."""
    o = NoisyOR(["A", "B"], [2, 2])
    tbl = o.get_table()
    assert tbl.shape == (2, 2, 2)
    # default leak inhibition is 1: the child is always OFF
    assert np.allclose(tbl[..., 0], 1)
    assert np.allclose(tbl[..., 1], 0)


def test_noisy_or_two_parents():
    """Checks the full table against hand computed values.

    Every parent which is ON multiplies the probability of the child
    being OFF by its inhibition probability."""
    o = NoisyOR(["A", "B"], [2, 2])
    o.add_variable("true", 0.9)
    o.add_variable("A", 0.2)
    o.add_variable("B", 0.5)
    tbl = o.get_table()

    expected_off = np.array([[0.9,       0.9 * 0.5],
                             [0.9 * 0.2, 0.9 * 0.2 * 0.5]])
    assert np.allclose(tbl[..., 0], expected_off)
    assert np.allclose(tbl[..., 1], 1 - expected_off)
    assert np.allclose(tbl.sum(axis=-1), 1)


def test_noisy_or_deterministic_or():
    """With zero inhibitions and no leak the gate is a logical OR."""
    o = NoisyOR(["A", "B"], [2, 2])
    o.add_variable("A", 0.0)
    o.add_variable("B", 0.0)
    tbl = o.get_table()
    for a, b in np.ndindex(2, 2):
        assert tbl[a, b, 1] == (1.0 if (a or b) else 0.0)


def test_noisy_or_variable_not_listed():
    """Parents without an inhibition probability do not influence the output."""
    o = NoisyOR(["C3", "C2"], [2, 2])
    o.add_variable("true", 0.25)
    o.add_variable("C2", 0.1)
    tbl = o.get_table()
    # independent of C3
    assert np.allclose(tbl[0], tbl[1])
    assert tbl[0, 0, 0] == pytest.approx(0.25)
    assert tbl[0, 1, 0] == pytest.approx(0.025)


def test_noisy_or_multivalued_parent():
    """For a multi-valued parent only the last value is ON."""
    o = NoisyOR(["A"], [3])
    o.add_variable("A", 0.4)
    tbl = o.get_table()
    assert tbl.shape == (3, 2)
    assert np.allclose(tbl[:, 0], [1.0, 1.0, 0.4])
    assert np.allclose(tbl[:, 1], [0.0, 0.0, 0.6])


def test_noisy_or_from_hugin_file(tmp_path):
    """NoisyOR potentials in Hugin files are converted into CPTs."""
    net_file = tmp_path / "noisy_or.net"
    net_file.write_text("""
net
{
}

node A
{
  states = ("off" "on");
}

node B
{
  states = ("off" "on");
}

node Y
{
  states = ("off" "on");
}

potential (A)
{
  data = (0.3 0.7);
}

potential (B)
{
  data = (0.6 0.4);
}

potential (Y | A B)
{
  model_data = ( NoisyOR(true, 0.9, A, 0.2, B, 0.5) );
}
""")
    bn = read_Hugin_file(str(net_file))
    y = bn['Y']
    assert y.is_noisy_or()
    assert [bn[i].name for i in y.parents] == ['A', 'B']
    expected_off = np.array([[0.9,       0.9 * 0.5],
                             [0.9 * 0.2, 0.9 * 0.2 * 0.5]])
    assert np.allclose(y.distr[..., 0], expected_off)
    assert np.allclose(y.distr[..., 1], 1 - expected_off)
    bn.validate()


@pytest.fixture
def noisy_bn():
    bn = BayesNet("noisy", [Attr('A', "CATEG", ["off", "on"]),
                            Attr('B', "CATEG", ["off", "on"]),
                            Attr('Y', "CATEG", ["off", "on"])])
    bn.addEdge('A', 'Y')
    bn.addEdge('B', 'Y')
    bn['Y'].set_noisy_or(0.9, {'A': 0.2, 'B': 0.5})
    return bn


def test_node_noisy_or(noisy_bn):
    bn = noisy_bn
    assert bn['Y'].is_noisy_or()
    # parents of Y are stored as [B, A] (addEdge inserts at the front)
    assert [bn[i].name for i in bn['Y'].parents] == ['B', 'A']
    expected_off = np.array([[0.9,       0.9 * 0.2],
                             [0.9 * 0.5, 0.9 * 0.5 * 0.2]])
    assert np.allclose(bn['Y'].distr[..., 0], expected_off)
    bn.validate()


def test_node_noisy_or_errors(noisy_bn):
    bn = BayesNet("x", [Attr('Z', "CATEG", [0, 1, 2])])
    with pytest.raises(ValueError):
        bn['Z'].set_noisy_or(0.5)
    with pytest.raises(ValueError):
        noisy_bn['A'].set_noisy_or(0.5, {'B': 0.1})  # B not a parent
    with pytest.raises(ValueError):
        noisy_bn['Y'].set_noisy_or(1.5)


def test_node_noisy_or_edge_changes(noisy_bn):
    bn = noisy_bn
    bn.delEdge('B', 'Y')
    assert bn['Y'].is_noisy_or()
    assert np.allclose(bn['Y'].distr[:, 0], [0.9, 0.9 * 0.2])
    bn.addEdge('B', 'Y')  # new parent has no influence
    assert bn['Y'].distr.shape == (2, 2, 2)
    assert np.allclose(bn['Y'].distr[0], bn['Y'].distr[1])
    bn['Y'].set_cpt()
    assert not bn['Y'].is_noisy_or()


def test_learning_keeps_noisy_or(noisy_bn):
    from BNInter.BayesNet.BayesNetLearn import learnProbabilitiesFromData
    bn = noisy_bn
    before = bn['Y'].distr.copy()
    data = [[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0]]
    learnProbabilitiesFromData(bn, data, priorN = 0)
    assert np.allclose(bn['Y'].distr, before)
    assert np.allclose(bn['A'].distr, [0.5, 0.5])


def test_noisy_or_hugin_round_trip(noisy_bn, tmp_path):
    fname = tmp_path / "out.net"
    with open(fname, "w") as f:
        write_Hugin_file(noisy_bn, f)
    bn2 = read_Hugin_file(str(fname))
    assert bn2['Y'].is_noisy_or()
    assert bn2['Y'].noisy_or['leak'] == pytest.approx(0.9)
    assert np.allclose(bn2['Y'].distr, noisy_bn['Y'].distr)
