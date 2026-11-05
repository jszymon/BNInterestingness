import numpy as np
import pytest

from BNInter.BayesNet import BayesNet, read_Hugin_file, write_Hugin_file
from BNInter.BayesNet.BayesNetLearn import learnProbabilitiesFromData, nodeCounts
from BNInter.BayesNet.LogisticCPT import logistic_cpt
from BNInter.DataAccess import Attr


def test_one_parent_unregularized_equals_mle():
    """With a single parent the model is saturated: without
    regularization it reproduces relative frequencies."""
    counts = np.array([[30, 10], [5, 15]])
    for std in [True, False]:
        cpt = logistic_cpt(counts, C=np.inf, standardize=std)
        assert np.allclose(cpt, [[0.75, 0.25], [0.25, 0.75]], atol=1e-4)


def test_regularization_shrinks_towards_marginal():
    counts = np.array([[30, 10], [5, 15]])
    free = logistic_cpt(counts, C=np.inf)
    reg = logistic_cpt(counts, C=0.001)
    marginal = counts.sum(axis=0) / counts.sum()
    assert np.allclose(reg.sum(axis=-1), 1)
    assert np.all(np.abs(reg - marginal) < np.abs(free - marginal))


def test_two_parents_additive_model():
    """Logistic regression has no interaction term, so with an
    interaction in the data the CPT differs from the MLE, but each row
    is still a distribution and the marginal fit is preserved."""
    counts = np.array([[[40, 10], [10, 40]],
                       [[10, 40], [45, 5]]], dtype=float)
    cpt = logistic_cpt(counts)
    assert cpt.shape == (2, 2, 2)
    assert np.allclose(cpt.sum(axis=-1), 1)
    # unregularized logistic regression matches the observed totals
    # of each class (score equations for the intercept)
    n = counts.sum(axis=-1)
    assert np.allclose((cpt * n[..., None]).sum(axis=(0, 1)),
                       counts.sum(axis=(0, 1)), atol=1e-2)


def test_multiclass_and_unobserved_class():
    counts = np.array([[10, 0, 5], [2, 0, 8], [0, 0, 0]])
    cpt = logistic_cpt(counts)
    assert cpt.shape == (3, 3)
    assert np.allclose(cpt.sum(axis=-1), 1)
    assert np.allclose(cpt[:, 1], 0)  # never observed class


def test_degenerate_cases():
    assert np.allclose(logistic_cpt(np.zeros((2, 3))), 1 / 3)
    assert np.allclose(logistic_cpt(np.array([3, 1])), [0.75, 0.25])
    assert np.allclose(logistic_cpt(np.array([[3, 0], [1, 0]])), [[1, 0], [1, 0]])


@pytest.fixture
def lr_bn():
    bn = BayesNet("lr", [Attr('A', "CATEG", [0, 1]),
                         Attr('B', "CATEG", [0, 1, 2]),
                         Attr('Y', "CATEG", [0, 1])])
    bn.addEdge('A', 'Y')
    bn.addEdge('B', 'Y')
    bn['Y'].set_logistic()
    return bn


def make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 3, n)
    p = 1 / (1 + np.exp(-(-1 + 1.5 * a + 0.7 * b)))
    y = (rng.random(n) < p).astype(int)
    return [list(r) for r in zip(a.tolist(), b.tolist(), y.tolist())]


def test_node_settings(lr_bn):
    y = lr_bn['Y']
    assert y.distr_type() == 'Logistic'
    assert y.logistic == {'C': float("inf"), 'standardize': True}
    with pytest.raises(ValueError):
        y.set_logistic(C=0)
    with pytest.raises(ValueError):
        y.set_logistic(C=-float("inf"))
    y.set_noisy_or(0.5)
    assert y.distr_type() == 'NoisyOR'
    y.set_logistic(1.0, False)
    assert y.distr_type() == 'Logistic'
    y.set_cpt()
    assert y.distr_type() == 'CPT'


def test_learning(lr_bn):
    data = make_data()
    learnProbabilitiesFromData(lr_bn, data, priorN=0)
    y = lr_bn['Y']
    assert y.distr_type() == 'Logistic'
    expected = logistic_cpt(nodeCounts(lr_bn, 2, data))
    assert np.allclose(y.distr, expected)
    lr_bn.validate()
    # P(Y=1) increases with A and B (parents are stored as [B, A])
    p1 = y.distr[..., 1]
    assert np.all(np.diff(p1, axis=0) > 0)
    assert np.all(np.diff(p1, axis=1) > 0)


def test_hugin_round_trip(lr_bn, tmp_path):
    lr_bn['Y'].set_logistic(0.5, False)
    learnProbabilitiesFromData(lr_bn, make_data(), priorN=0)
    fname = tmp_path / "lr.net"
    with open(fname, "w") as f:
        write_Hugin_file(lr_bn, f)
    bn2 = read_Hugin_file(str(fname))
    assert bn2['Y'].logistic == {'C': 0.5, 'standardize': False}
    assert np.allclose(bn2['Y'].distr, lr_bn['Y'].distr)
    assert bn2['A'].distr_type() == 'CPT'
