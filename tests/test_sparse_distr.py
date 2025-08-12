import numpy as np

from pytest import approx

from BNInter.Utils import SparseDistr

def test_sparse_distr_unif():
    s = SparseDistr((5,5))
    assert "Uniform prob: 0.04" in str(s)
    assert np.allclose(s.to_array(), 1/25)
    assert s.P((1,1)) == approx(1/25)
    assert s.P((2,2)) == approx(1/25)
    s = s.sample(10)
    assert np.all(s>=0)
    assert np.all(s<5)

def test_sparse_distr_unif():
    s = SparseDistr((5,5), {(1,1): 1.0})
    assert "remaining cells: 0.0" in str(s)
    expeced = np.zeros((5,5))
    expeced[1][1] = 1
    assert np.allclose(s.to_array(), expeced)
    assert s.P((1,1)) == approx(1)
    assert s.P((2,2)) == approx(0)
    s = s.sample(10)
    assert np.all(s==1)

def test_sparse_distr_w_prior():
    s = SparseDistr((5,5), {(1,1): 0.5, (2,2): 0.5}, prior_factor=0.1)
    assert "remaining cells: 0.004" in str(s)
    assert s.to_array().sum() == approx(1)
    assert s.P((0,0)) == approx(0.1*1/25)
    assert s.P((1,1)) == approx(0.9*0.5 + 0.1*1/25)
    assert s.P((2,2)) == approx(0.9*0.5 + 0.1*1/25)
    n = 1_000_000
    x = s.sample(n)
    assert np.all(x>=0)
    assert np.all(x<5)
    x = x.tolist()
    p0 = 0.1*1/25
    p1 = 0.9*0.5 + 0.1*1/25
    assert abs(sum(r==[0,0] for r in x) / n - p0) <= 0.1
    assert abs(sum(r==[1,1] for r in x) / n - p1) <= 0.1
    assert abs(sum(r==[2,2] for r in x) / n - p1) <= 0.1
