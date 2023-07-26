# cdf/inverse_normal.c
# based on GSL routine


import math
try:
    import pygsl.sf
    have_gsl = True
except:
    have_gsl = False


# Computes the inverse normal cumulative distribution function 
# according to the algorithm shown in 
#
#     Wichura, M.J. (1988).
#     Algorithm AS 241: The Percentage Points of the Normal Distribution.
#     Applied Statistics, 37, 477-484.
#


def rat_eval(a, na, b, nb, x):
    u = a[na - 1]
    i = na - 1
    while i > 0:
        u = x * u + a[i - 1]
        i -= 1

    v = b[nb - 1]
    j = nb - 1
    while j > 0:
        v = x * v + b[j - 1]
        j -= 1
    
    r = u / v
    return r



def small(q):
    a = [ 3.387132872796366608, 133.14166789178437745,
          1971.5909503065514427, 13731.693765509461125,
          45921.953931549871457, 67265.770927008700853,
          33430.575583588128105, 2509.0809287301226727 ]
    b = [ 1.0, 42.313330701600911252,
          687.1870074920579083, 5394.1960214247511077,
          21213.794301586595867, 39307.89580009271061,
          28729.085735721942674, 5226.495278852854561 ]
    
    r = 0.180625 - q * q
    x = q * rat_eval(a, 8, b, 8, r)
    return x


def intermediate(r):
    a = [ 1.42343711074968357734, 4.6303378461565452959,
          5.7694972214606914055, 3.64784832476320460504,
          1.27045825245236838258, 0.24178072517745061177,
          0.0227238449892691845833, 7.7454501427834140764e-4]
    b = [ 1.0, 2.05319162663775882187,
          1.6763848301838038494, 0.68976733498510000455,
          0.14810397642748007459, 0.0151986665636164571966,
          5.475938084995344946e-4, 1.05075007164441684324e-9]
    x = rat_eval(a, 8, b, 8, (r - 1.6))
    return x


def tail(r):
    a = [ 6.6579046435011037772, 5.4637849111641143699,
          1.7848265399172913358, 0.29656057182850489123,
          0.026532189526576123093, 0.0012426609473880784386,
          2.71155556874348757815e-5, 2.01033439929228813265e-7]
    b = [ 1.0, 0.59983220655588793769,
          0.13692988092273580531, 0.0148753612908506148525,
          7.868691311456132591e-4, 1.8463183175100546818e-5,
          1.4215117583164458887e-7, 2.04426310338993978564e-15]
    x = rat_eval (a, 8, b, 8, (r - 5.0))
    return x

def cdf_ugaussian_Pinv(P):
    dP = P - 0.5
    if P == 1.0:
        return float('Inf')
    elif P == 0.0:
        return float('-Inf')

    if abs(dP) <= 0.425:
        return small(dP)

    if P < 0.5:
        pp = P
    else:
        pp = 1.0 - P
        
    r = math.sqrt(-math.log(pp))

    if r <= 5.0:
        x = intermediate(r)
    else:
        x = tail(r)

    if P < 0.5:
        return -x
    else:
        return x


def cdf_ugaussian_Qinv(Q):
    dQ = Q - 0.5;

    if Q == 1.0:
        return float('-Inf')
    elif Q == 0.0:
        return float('Inf')

    if abs(dQ) <= 0.425:
        x = small(dQ)
        return -x

    if Q < 0.5:
        pp = Q
    else:
        pp = 1.0 - Q

    r = math.sqrt(-math.log(pp))

    if r <= 5.0:
        x = intermediate(r)
    else:
        x = tail(r)

    if Q < 0.5:
        return x
    else:
        return -x


def cdf_gaussian_Pinv(P, sigma):
    return sigma * cdf_ugaussian_Pinv(P)

def cdf_gaussian_Qinv(Q, sigma):
    return sigma * cdf_ugaussian_Qinv(Q)


GSL_DBL_EPSILON = 1e-10
TEST_TOL0 = (2.0*GSL_DBL_EPSILON)
TEST_TOL1 = (16.0*GSL_DBL_EPSILON)
TEST_TOL2 = (256.0*GSL_DBL_EPSILON)
TEST_TOL3 = (2048.0*GSL_DBL_EPSILON)
TEST_TOL4 = (16384.0*GSL_DBL_EPSILON)
TEST_TOL5 = (131072.0*GSL_DBL_EPSILON)
TEST_TOL6 = (1048576.0*GSL_DBL_EPSILON)



def TEST(func, arg, value, tol):
    res = func(arg)
    if(abs(value - res) > tol):
        print("error", arg)


if __name__ == "__main__":
    
    TEST (cdf_ugaussian_Pinv, (0.9999997133), 5.0, 1e-4)
    TEST (cdf_ugaussian_Pinv, (0.9999683288), 4.0, 1e-6)
    TEST (cdf_ugaussian_Pinv, (0.9986501020), 3.0, 1e-8)
    TEST (cdf_ugaussian_Pinv, (0.977249868051821), 2.0, 1e-14)
    TEST (cdf_ugaussian_Pinv, (0.841344746068543), 1.0, TEST_TOL3)
    TEST (cdf_ugaussian_Pinv, (0.691462461274013), 0.5, TEST_TOL2)
    TEST (cdf_ugaussian_Pinv, (0.655421741610324), 0.4, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (0.617911422188953), 0.3, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (0.579259709439103), 0.2, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (0.539827837277029), 0.1, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (0.5), 0.0, TEST_TOL0)
    TEST (cdf_ugaussian_Pinv, (4.60172162722971e-1), -0.1, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (4.20740290560897e-1), -0.2, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (3.82088577811047e-1), -0.3, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (3.44578258389676e-1), -0.4, TEST_TOL1)
    TEST (cdf_ugaussian_Pinv, (3.08537538725987e-1), -0.5, TEST_TOL2)
    TEST (cdf_ugaussian_Pinv, (1.58655253931457e-1), -1.0, TEST_TOL3)
    TEST (cdf_ugaussian_Pinv, (2.2750131948179e-2), -2.0, 1e-14)
    TEST (cdf_ugaussian_Pinv, (1.349898e-3), -3.0, 1e-8)
    TEST (cdf_ugaussian_Pinv, (3.16712e-5), -4.0, 1e-6)
    TEST (cdf_ugaussian_Pinv, (2.86648e-7), -5.0, 1e-4)

    TEST (cdf_ugaussian_Pinv, (7.61985302416052e-24), -10.0, 1e-4)

    TEST (cdf_ugaussian_Qinv, (7.61985302416052e-24), 10.0, 1e-4)

    TEST (cdf_ugaussian_Qinv, (2.86648e-7), 5.0, 1e-4)
    TEST (cdf_ugaussian_Qinv, (3.16712e-5), 4.0, 1e-6)
    TEST (cdf_ugaussian_Qinv, (1.349898e-3), 3.0, 1e-8)
    TEST (cdf_ugaussian_Qinv, (2.2750131948179e-2), 2.0, 1e-14)
    TEST (cdf_ugaussian_Qinv, (1.58655253931457e-1), 1.0, TEST_TOL3)
    TEST (cdf_ugaussian_Qinv, (3.08537538725987e-1), 0.5, TEST_TOL2)
    TEST (cdf_ugaussian_Qinv, (3.44578258389676e-1), 0.4, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (3.82088577811047e-1), 0.3, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (4.20740290560897e-1), 0.2, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (4.60172162722971e-1), 0.1, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (0.5), 0.0, TEST_TOL0)
    TEST (cdf_ugaussian_Qinv, (0.539827837277029), -0.1, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (0.579259709439103), -0.2, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (0.617911422188953), -0.3, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (0.655421741610324), -0.4, TEST_TOL1)
    TEST (cdf_ugaussian_Qinv, (0.691462461274013), -0.5, TEST_TOL2)
    TEST (cdf_ugaussian_Qinv, (0.841344746068543), -1.0, TEST_TOL3)
    TEST (cdf_ugaussian_Qinv, (0.977249868051821), -2.0, 1e-14)
    TEST (cdf_ugaussian_Qinv, (0.9986501020), -3.0, 1e-8)
    TEST (cdf_ugaussian_Qinv, (0.9999683288), -4.0, 1e-6)
    TEST (cdf_ugaussian_Qinv, (0.9999997133), -5.0, 1e-4)
