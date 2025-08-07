# cdf/gauss.c
# based on GSL routine

import math

# Computes the cumulative distribution function for the Gaussian
# distribution using a rational function approximation.  The
# computation is for the standard Normal distribution, i.e., mean 0
# and standard deviation 1. If you want to compute Pr(X < t) for a
# Gaussian random variable X with non-zero mean m and standard
# deviation sd not equal to 1, find gsl_cdf_ugaussian ((t-m)/sd).
# This approximation is accurate to at least double precision. The
# accuracy was verified with a pari-gp script.  The largest error
# found was about 1.4E-20. The coefficients were derived by Cody.
# 
# References:
# 
# W.J. Cody. "Rational Chebyshev Approximations for the Error
# Function," Mathematics of Computation, v23 n107 1969, 631-637.
# 
# W. Fraser, J.F Hart. "On the Computation of Rational Approximations
# to Continuous Functions," Communications of the ACM, v5 1962.
# 
# W.J. Kennedy Jr., J.E. Gentle. "Statistical Computing." Marcel Dekker. 1980.


#M_1_SQRT2PI (M_2_SQRTPI * M_SQRT1_2 / 2.0)
#define SQRT32 (4.0 * M_SQRT2)


M_SQRT2 = math.sqrt(2.0)
M_SQRT1_2 = math.sqrt(0.5)
M_2_SQRTPI = 2.0 / math.sqrt(math.pi)
M_1_SQRT2PI = (M_2_SQRTPI * M_SQRT1_2 / 2.0)
SQRT32 = (4.0 * M_SQRT2)


GSL_DBL_EPSILON = 1e-10
GAUSS_EPSILON = (GSL_DBL_EPSILON / 2)
GAUSS_XUPPER = (8.572)
GAUSS_XLOWER = (-37.519)

GAUSS_SCALE = (16.0)

def get_del (x, rational):
    xsq = math.floor (x * GAUSS_SCALE) / GAUSS_SCALE
    dl = (x - xsq) * (x + xsq)
    dl *= 0.5
    
    result = math.exp (-0.5 * xsq * xsq) * math.exp (-1.0 * dl) * rational
    return result

# 
# Normal cdf for fabs(x) < 0.66291
# 
def gauss_small (x):
    a = [2.2352520354606839287,
         161.02823106855587881,
         1067.6894854603709582,
         18154.981253343561249,
         0.065682337918207449113]
    b = [47.20258190468824187,
         976.09855173777669322,
         10260.932208618978205,
         45507.789335026729956]
  
    xsq = x * x
    xnum = a[4] * xsq
    xden = xsq
  
    for i in range(4):
        xnum = (xnum + a[i]) * xsq
        xden = (xden + b[i]) * xsq
  
    result = x * (xnum + a[3]) / (xden + b[3])
    return result

# 
# Normal cdf for 0.66291 < fabs(x) < sqrt(32).
# 
def gauss_medium (x):
    c = [0.39894151208813466764,
         8.8831497943883759412,
         93.506656132177855979,
         597.27027639480026226,
         2494.5375852903726711,
         6848.1904505362823326,
         11602.651437647350124,
         9842.7148383839780218,
         1.0765576773720192317e-8]

    d = [22.266688044328115691,
         235.38790178262499861,
         1519.377599407554805,
         6485.558298266760755,
         18615.571640885098091,
         34900.952721145977266,
         38912.003286093271411,
         19685.429676859990727]
  
    absx = abs (x)
  
    xnum = c[8] * absx
    xden = absx
  
    for i in range(7):
        xnum = (xnum + c[i]) * absx
        xden = (xden + d[i]) * absx
  
    temp = (xnum + c[7]) / (xden + d[7])
    result = get_del (x, temp)
    return result


# 
# Normal cdf for 
# {sqrt(32) < x < GAUSS_XUPPER} union { GAUSS_XLOWER < x < -sqrt(32) }.
# 
def gauss_large (x):
    p = [0.21589853405795699,
         0.1274011611602473639,
         0.022235277870649807,
         0.001421619193227893466,
         2.9112874951168792e-5,
         0.02307344176494017303]

    q = [1.28426009614491121,
         0.468238212480865118,
         0.0659881378689285515,
         0.00378239633202758244,
         7.29751555083966205e-5]
  
    absx = abs (x)
    xsq = 1.0 / (x * x)
    xnum = p[5] * xsq
    xden = xsq
  
    for i in range(4):
        xnum = (xnum + p[i]) * xsq
        xden = (xden + q[i]) * xsq
  
    temp = xsq * (xnum + p[4]) / (xden + q[4])
    temp = (M_1_SQRT2PI - temp) / absx
    result = get_del (x, temp)
    return result


def cdf_ugaussian_P (x):
    absx = abs (x)
  
    if absx < GAUSS_EPSILON:
        result = 0.5
        return result
    elif absx < 0.66291:
        result = 0.5 + gauss_small (x)
        return result
    elif absx < SQRT32:
        result = gauss_medium (x)
        if x > 0.0:
            result = 1.0 - result
        return result
    elif x > GAUSS_XUPPER:
        result = 1.0
        return result
    elif x < GAUSS_XLOWER:
        result = 0.0
        return result
    else:
        result = gauss_large (x)
        if x > 0.0:
            result = 1.0 - result
  
    return result


def cdf_ugaussian_Q (x):
    absx = abs (x)
  
    if absx < GAUSS_EPSILON:
        result = 0.5
        return result
    elif absx < 0.66291:
        result = gauss_small (x) 
        if x < 0.0:
            result = abs (result) + 0.5
        else:
            result = 0.5 - result
        return result
    elif absx < SQRT32:
        result = gauss_medium (x)
        if x < 0.0:
            result = 1.0 - result  
        return result
    elif x > -(GAUSS_XLOWER):
        result = 0.0
        return result
    elif x < -(GAUSS_XUPPER):
        result = 1.0
        return result
    else:
        result = gauss_large (x)
        if x < 0.0:
            result = 1.0 - result
  
    return result


def cdf_gaussian_P (x, sigma):
    return cdf_ugaussian_P (x / sigma)


def cdf_gaussian_Q (x, sigma):
    return cdf_ugaussian_Q (x / sigma)



TEST_TOL0 = (2.0*GSL_DBL_EPSILON)
TEST_TOL1 = (16.0*GSL_DBL_EPSILON)
TEST_TOL2 = (256.0*GSL_DBL_EPSILON)
TEST_TOL3 = (2048.0*GSL_DBL_EPSILON)
TEST_TOL4 = (16384.0*GSL_DBL_EPSILON)
TEST_TOL5 = (131072.0*GSL_DBL_EPSILON)
TEST_TOL6 = (1048576.0*GSL_DBL_EPSILON)


def TEST(func, arg, value, tol):
    res = func(*arg)
    if(abs(value - res) > tol):
        print("error", arg, value, res, tol)


if __name__ == "__main__":
    TEST(cdf_gaussian_P, (-1.0000000000000000e+10,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+09,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+08,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+07,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+06,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+05,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+04,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+03,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+02,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+01,1.3), 7.225229227927e-15, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e+00,1.3), 2.208781637125e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000001e-01,1.3), 4.693423696034e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e-02,1.3), 4.969312434916e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e-03,1.3), 4.996931213530e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e-04,1.3), 4.999693121323e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000001e-05,1.3), 4.999969312132e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-9.9999999999999995e-07,1.3), 4.999996931213e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-9.9999999999999995e-08,1.3), 4.999999693121e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e-08,1.3), 4.999999969312e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000001e-09,1.3), 4.999999996931e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (-1.0000000000000000e-10,1.3), 4.999999999693e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (0.0000000000000000e+00,1.3), 5.000000000000e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e-10,1.3), 5.000000000307e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000001e-09,1.3), 5.000000003069e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e-08,1.3), 5.000000030688e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (9.9999999999999995e-08,1.3), 5.000000306879e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (9.9999999999999995e-07,1.3), 5.000003068787e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000001e-05,1.3), 5.000030687868e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e-04,1.3), 5.000306878677e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e-03,1.3), 5.003068786470e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e-02,1.3), 5.030687565084e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000001e-01,1.3), 5.306576303966e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+00,1.3), 7.791218362875e-01, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+01,1.3), 1.000000000000e-00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+02,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+03,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+04,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+05,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+06,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+07,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+08,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+09,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_P, (1.0000000000000000e+10,1.3), 1.000000000000e+00, TEST_TOL6)

    TEST(cdf_gaussian_Q, (1.0000000000000000e+10,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+09,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+08,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+07,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+06,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+05,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+04,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+03,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+02,1.3), 0.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+01,1.3), 7.225229227927e-15, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e+00,1.3), 2.208781637125e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000001e-01,1.3), 4.693423696034e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e-02,1.3), 4.969312434916e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e-03,1.3), 4.996931213530e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e-04,1.3), 4.999693121323e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000001e-05,1.3), 4.999969312132e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (9.9999999999999995e-07,1.3), 4.999996931213e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (9.9999999999999995e-08,1.3), 4.999999693121e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e-08,1.3), 4.999999969312e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000001e-09,1.3), 4.999999996931e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (1.0000000000000000e-10,1.3), 4.999999999693e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (0.0000000000000000e+00,1.3), 5.000000000000e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e-10,1.3), 5.000000000307e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000001e-09,1.3), 5.000000003069e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e-08,1.3), 5.000000030688e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-9.9999999999999995e-08,1.3), 5.000000306879e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-9.9999999999999995e-07,1.3), 5.000003068787e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000001e-05,1.3), 5.000030687868e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e-04,1.3), 5.000306878677e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e-03,1.3), 5.003068786470e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e-02,1.3), 5.030687565084e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000001e-01,1.3), 5.306576303966e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+00,1.3), 7.791218362875e-01, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+01,1.3), 1.000000000000e-00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+02,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+03,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+04,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+05,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+06,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+07,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+08,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+09,1.3), 1.000000000000e+00, TEST_TOL6)
    TEST(cdf_gaussian_Q, (-1.0000000000000000e+10,1.3), 1.000000000000e+00, TEST_TOL6)
    
