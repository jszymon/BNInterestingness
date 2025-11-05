import numpy as np
import pytest

from BNInter.BayesNet import BayesNet
from BNInter.DataAccess import Attr

@pytest.fixture
def basic_bayes_net():
    """Provides a basic BayesNet instance for testing."""
    bn = BayesNet("testNet",
                  [Attr('A', "CATEG", [0, 1]),
                   Attr('B', "CATEG", [0, 1]),
                   Attr('Y', "CATEG", [0, 1, 2])])
    return bn

def test_initial_bayes_net_setup(basic_bayes_net):
    """Tests the initial setup and string representation of the BayesNet."""
    bn = basic_bayes_net
    assert str(bn) == "Bayesian network: testNet\nNodes:\nA:[0, 1]\nParents: []\nDistribution:\n()->[0.5 0.5]\n\nB:[0, 1]\nParents: []\nDistribution:\n()->[0.5 0.5]\n\nY:[0, 1, 2]\nParents: []\nDistribution:\n()->[0.333333 0.333333 0.333333]\n"
    assert 'A' in bn.names_2_numbers_map
    assert 'B' in bn.names_2_numbers_map
    assert 'Y' in bn.names_2_numbers_map

def test_set_parents_distribution(basic_bayes_net):
    """Tests setting parent distributions and validation."""
    bn = basic_bayes_net
    bn['Y'].set_parents_distr(['B'], np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 0.2]]))
    
    # Verify the distribution has been set
    assert np.array_equal(bn['Y'].distr, np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 0.2]]))
    assert 'B' in [bn[i].name for i in bn['Y'].parents]

    # Test validation (should pass with the given distribution)
    try:
        bn.validate()
    except ValueError:
        pytest.fail("bn.validate() raised ValueError unexpectedly")

    # Test with a distribution that sums incorrectly to ensure validate catches it
    bn['Y'].set_parents_distr(['B'], np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 1.2]]))
    with pytest.raises(ValueError, match="Conditional distribution for Y does not sum to 1.0 for all parent combinations"):
        bn.validate()

def test_probability_calculation(basic_bayes_net):
    """Tests the calculation of joint probability P([0,0,0])."""
    bn = basic_bayes_net
    bn['Y'].set_parents_distr(['B'], np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 0.2]]))
    
    # A, B are independent since the is no edge between them
    # P(A=0) = 0.5, P(B=0) = 0.5, P(Y=0|B=0) = 0.7
    # P(A=0, B=0, Y=0) = P(A=0) * P(B=0) * P(Y=0|B=0) = 0.5 * 0.5 * 0.7 = 0.175
    assert bn.P([0, 0, 0]) == pytest.approx(0.175)
    assert bn.P([1, 1, 2]) == pytest.approx(0.05)

def test_joint_probability_distribution(basic_bayes_net):
    """Tests the generation of the joint probability distribution string."""
    bn = basic_bayes_net
    bn['Y'].set_parents_distr(['B'], np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 0.2]]))
    
    # Since A and B have default uniform distributions (0.5 for each state)
    # and there is no edge between A, B, the joint distribution will be:
    # P(A,B,Y) = P(A) * P(B) * P(Y|B)
    
    # Expected joint probabilities:
    # P(0,0,0) = 0.5 * 0.5 * 0.7 = 0.175
    # P(0,0,1) = 0.5 * 0.5 * 0.1 = 0.025
    # P(0,0,2) = 0.5 * 0.5 * 0.2 = 0.05
    # P(0,1,0) = 0.5 * 0.5 * 0.5 = 0.125
    # P(0,1,1) = 0.5 * 0.5 * 0.3 = 0.075
    # P(0,1,2) = 0.5 * 0.5 * 0.2 = 0.05
    # P(1,0,0) = 0.5 * 0.5 * 0.7 = 0.175
    # P(1,0,1) = 0.5 * 0.5 * 0.1 = 0.025
    # P(1,0,2) = 0.5 * 0.5 * 0.2 = 0.05
    # P(1,1,0) = 0.5 * 0.5 * 0.5 = 0.125
    # P(1,1,1) = 0.5 * 0.5 * 0.3 = 0.075
    # P(1,1,2) = 0.5 * 0.5 * 0.2 = 0.05
    
    joint_p_array = bn.jointP()
    # Check a few specific values, or the sum
    assert np.isclose(joint_p_array[0,0,0], 0.175)
    assert np.isclose(joint_p_array[0,0,1], 0.025)
    assert np.isclose(joint_p_array[0,0,2], 0.05)
    assert np.isclose(joint_p_array.sum(), 1.0) # Total probability should sum to 1


def test_add_and_delete_edge(basic_bayes_net):
    """Tests adding and deleting edges between nodes."""
    bn = basic_bayes_net
    
    # Add edge
    bn.addEdge('A', 'B')
    assert 'A' in [bn[i].name for i in bn['B'].parents]
    assert "B:[0, 1]\nParents: ['A']" in str(bn)

    # Delete edge
    bn.delEdge('A', 'B')
    assert 'A' not in [bn[i].name for i in bn['B'].parents]
    assert "B:[0, 1]\nParents: ['A']" not in str(bn)

def test_delete_all_parents(basic_bayes_net):
    """Tests deleting all parents of a node."""
    bn = basic_bayes_net
    bn['Y'].set_parents_distr(['B'], np.array([[0.7, 0.1, 0.2], [0.5, 0.3, 0.2]]))
    assert 'B' in [bn[i].name for i in bn['Y'].parents]

    bn['Y'].del_all_parents()
    assert not bn['Y'].parents
    # After deleting parents, the distribution should revert to a default (e.g., uniform for 
    # a node with no parents) or be explicitly set to indicate no dependencies.
    # We'll check if the distribution array matches the expected size for no parents
    assert bn['Y'].distr.shape == (3,) # Y has 3 states, so a 1D array of length 3 is expected for no parents.
    assert np.allclose(bn['Y'].distr, [1/3, 1/3, 1/3]) # Default is uniform distribution
