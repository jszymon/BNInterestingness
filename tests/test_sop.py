import numpy
import pytest
from BNInter import sop

@pytest.fixture
def simple_sop_instance():
    """Provides a configured simple_sop instance for testing."""
    thesop = sop.simple_sop(3)
    dist1 = ([0, 1], numpy.array([[1.0 / 9, 2.0 / 9, 0.0 / 9],
                                  [3.0 / 9, 0.0 / 9, 0.0 / 9],
                                  [1.0 / 9, 1.0 / 9, 1.0 / 9]]))
    dist2 = ([2, 1], numpy.array([[1.0 / 3, 0.0 / 3, 0.0 / 3],
                                  [1.0 / 3, 2.0 / 3, 1.0 / 3],
                                  [1.0 / 3, 1.0 / 3, 2.0 / 3]]))
    thesop.add_factor(*dist1)
    thesop.add_factor(*dist2)
    return thesop

@pytest.fixture
def array_sop_instance():
    """Provides a configured array_sop instance for testing."""
    asop = sop.array_sop(3)
    dist1 = ([0, 1], numpy.array([[1.0 / 9, 2.0 / 9, 0.0 / 9],
                                  [3.0 / 9, 0.0 / 9, 0.0 / 9],
                                  [1.0 / 9, 1.0 / 9, 1.0 / 9]]))
    dist2 = ([2, 1], numpy.array([[1.0 / 3, 0.0 / 3, 0.0 / 3],
                                  [1.0 / 3, 2.0 / 3, 1.0 / 3],
                                  [1.0 / 3, 1.0 / 3, 2.0 / 3]]))
    asop.add_factor(*dist1)
    asop.add_factor(*dist2)
    return asop

def test_simple_sop_computations(simple_sop_instance):
    """Tests various simple_sop compute calls with different ranges."""
    thesop = simple_sop_instance

    thesop.prepare([0, 0, 0], [2, 2, 2])
    assert thesop.compute() == pytest.approx(1.0)
    assert thesop.complexity() == 2
    assert thesop.cost() == (22, 25)
    assert thesop.mem_cost() == 0

    thesop.prepare([0, 0, 0], [2, 0, 2])
    assert thesop.compute() == pytest.approx(5/9)
    assert thesop.complexity() == 1
    assert thesop.cost() == (8, 9)
    assert thesop.mem_cost() == 0

    thesop.prepare([0, 1, 0], [2, 1, 2])
    assert thesop.compute() == pytest.approx(3/9)
    assert thesop.complexity() == 1
    assert thesop.cost() == (8, 9)
    assert thesop.mem_cost() == 0

    thesop.prepare([0, 2, 0], [2, 2, 2])
    assert thesop.compute() == pytest.approx(1/9)
    assert thesop.complexity() == 1
    assert thesop.cost() == (8, 9)
    assert thesop.mem_cost() == 0

def test_array_sop_computations(array_sop_instance):
    """Tests various array_sop compute calls with different preparations."""
    asop = array_sop_instance
    
    asop.prepare([1])
    assert asop.compute() == pytest.approx([5/9, 3/9, 1/9])
    assert asop.complexity() == 2
    assert asop.cost() == (12, 24)
    assert asop.mem_cost() == 144
    
    asop.prepare([])
    assert asop.compute() == pytest.approx(1.0)
    assert asop.complexity() == 2
    assert asop.cost() == (14, 25)
    assert asop.mem_cost() == 144
    
    asop.prepare([0, 1, 2])
    expected_result = numpy.array([[[1/27, 1/27, 1/27],
                                    [0.0, 4/27, 2/27],
                                    [0.0, 0.0, 0.0]],
                                   [[1/9, 1/9, 1/9],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0]],
                                   [[1/27, 1/27, 1/27],
                                    [0, 2/27, 1/27],
                                    [0, 1/27, 2/27]]])
    assert numpy.allclose(asop.compute(), expected_result)
    assert asop.complexity() == 3
    assert asop.cost() == (0, 54)
    assert asop.mem_cost() == 360
    
    asop.prepare([0, 1])
    expected_result = numpy.array([[1/9, 2/9, 0.0],
                                   [3/9, 0.0, 0.0],
                                   [1/9, 1/9, 1/9]])
    assert numpy.allclose(asop.compute(), expected_result)
    assert asop.complexity() == 2
    assert asop.cost() == (6, 27)
    assert asop.mem_cost() == 168

    asop.prepare([1, 0])
    expected_result = numpy.array([[1/9, 3/9, 1/9],
                                   [2/9, 0.0, 1/9],
                                   [0.0, 0.0, 1/9]])
    assert numpy.allclose(asop.compute(), expected_result)
    assert asop.complexity() == 2
    assert asop.cost() == (6, 27)
    assert asop.mem_cost() == 168
