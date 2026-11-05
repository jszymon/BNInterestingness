"""Functions related to learning Bayesian networks."""

import operator
import numpy as np
from math import exp
from math import lgamma
from functools import reduce

from ..Utils import compute_counts_array
from ..Utils import compute_counts_dict
from ..Utils import convert_counts_dicts2numpyarrays

from ..DataAccess import Attr
from .BayesNet import BayesNet
from .BNutils import blockiter, distr_2_str


def learnProbabilitiesFromData(bn, dataset, priorN = 1):
    asets = [n.parents + [i] for (i,n) in enumerate(bn) if not n.in_joint]
    counts, N, missing_counts = compute_counts_array(asets, bn.get_shape(), dataset)
    # single nodes
    for j, n in enumerate(bn):
        if n.in_joint:
            continue
        c = counts[tuple(n.parents + [j])]
        ri = len(n.domain)
        rdistr = np.ravel(n.distr)
        rcount = np.ravel(c)
        i = 0
        while i < len(rdistr):
            sum = 0
            for j in range(ri):
                sum += rcount[i + j]
            if sum > 0 or priorN > 0:
                for j in range(ri):
                    rdistr[i + j] = (priorN / ri + rcount[i + j]) / (priorN + sum)
            else:
                #print "Warning: no data about distribution, node ", n.attrname
                for j in range(ri):
                    rdistr[i + j] = 1.0 / ri
            i += ri
        rdistr.shape = n.distr.shape
        n.distr = rdistr
    # joint distribution nodes
    asets = [jn.nodes for jn in bn.joint_distrs]
    if hasattr(dataset, "rewind"): # allow lists as datasets
        dataset.rewind()
    counts, N, missing_counts = compute_counts_dict(asets, dataset)
    for jn in bn.joint_distrs:
        c = counts[tuple(jn.nodes)]
        n = N - missing_counts[tuple(jn.nodes)]
        alpha = jn.distr.size * priorN / (n + jn.distr.size * priorN)
        new_p = {x: v / n for x, v in c.items()}
        jn.distr.set_distr(new_p, prior_factor = alpha)

def BDeu(bn, dataset, priorN = None):
    """Compute the natural logarithm of probability that the
    dataset is generated from a network with current structure.

    Dataset should be a sequence of sequences of length matching
    the number of input variables.  priorN out trust in the prior
    distribution.  See Heckerman, B-Course site for details

    Joint distribution nodes are scored as a single variable without
    parents whose values are all combinations of member values.
    Their counts are kept sparse, empty cells contribute zero."""

    asets = [n.parents + [i] for (i,n) in enumerate(bn) if not n.in_joint]
    joint_asets = [jn.nodes for jn in bn.joint_distrs]
    # single pass over data, dense arrays only for ordinary nodes
    counts, N, missing_counts = compute_counts_dict(asets + joint_asets, dataset)
    shape = bn.get_shape()
    dense_counts = convert_counts_dicts2numpyarrays(shape, {tuple(a): counts[tuple(a)] for a in asets})


    if priorN is None:
        priorN = float(sum([len(n.domain) for n in bn])) / 2 / len(bn)
    priorN = float(priorN)

    lnP = 0
    for i, n in enumerate(bn):
        if n.in_joint:
            continue
        c = dense_counts[tuple(n.parents + [i])]
        ri = len(n.domain)
        qi = reduce(operator.mul, [len(bn[i].domain) for i in n.parents], 1)
        #print i, n.parentnumbers, c, ri, qi
        for nodeattrcounts in blockiter(np.ravel(c), ri): # for j = 1 to q_i
            Nij = sum(nodeattrcounts)
            #print "-->",nodeattrcounts
            lnP += lgamma(float(priorN) / qi) \
                   - lgamma(float(priorN) / qi + Nij)
            for Nijk in nodeattrcounts: # for k = 1 to r_i
                lnP += lgamma(float(priorN) / qi / ri + Nijk) \
                       - lgamma(float(priorN) / qi / ri)
    # joint distribution nodes: q_i = 1, r_i = product of domain sizes
    for jn in bn.joint_distrs:
        c = counts[tuple(jn.nodes)]
        ri = jn.distr.size
        Nij = sum(c.values())
        lnP += lgamma(priorN) - lgamma(priorN + Nij)
        for Nijk in c.values():
            lnP += lgamma(priorN / ri + Nijk) - lgamma(priorN / ri)
    return lnP

def makeIndependentStructure(bn):
    """Make all attributes in bn independent.

    I.e. remove all edges."""
    for node in bn:
        n = len(node.domain)
        node.set_parents_distr([], np.zeros(n) + 1.0/n)
