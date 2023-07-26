import itertools
import numpy

import sop.sop
import sop.cached_sop
from .DiscreteDistr import DiscreteDistrSOP

import Apriori
import Apriori.AprioriDistr
from .DDApriori import DDApriori


def create_sop(bn):
    """create sum of products for Bayesian network BN."""
    #asop = sop.sop.array_sop(bn.n)
    asop = sop.cached_sop.cached_array_sop(len(bn))
    for i, node in enumerate(bn):
        asop.add_factor(node.parents + [i], node.distr)
    return asop


# #compute interestingness from joint
# joint = bn.jointP()
# #print BayesNet.BayesNet.distr_2_str(joint)
# #print "sum=", numpy.sum(numpy.ravel(joint))
# 
# isets_w_inter = []
# for iset in itemsets:
#     # form the slice
#     sl = [slice(0,len(d)) for d in domains]
#     p = float(iset[1])/N # actual support
#     for a, v in iset[0]:
#         sl[a] = slice(v, v+1)
#     subdistr = joint[tuple(sl)].copy()
#     ep = numpy.sum(numpy.ravel(subdistr))
#     inter = abs(ep-p)
#     if inter > 0:
#         iset_str = [attrnames[i[0]] + "=" + str(i[1]) for i in iset[0]]
#         isets_w_inter.append((iset[0], inter))






class BN_interestingness_exact(object):
    """Class for encapsulating the algorithm for computing attrset
    interestingness exactly."""
    def __init__(self, bn, ds):
        self.bn = bn
        self.ds = ds


    def compute_frequent_attr_sets_in_data(self, minsup, maxK, apriori_debug):
        self.ds.rewind()
        self.apriori = Apriori.AprioriDistr.AprioriDistr(self.ds)
        self.apriori.minsup = minsup
        self.apriori.maxK = maxK
        self.apriori.debug = apriori_debug
        print("Running Apriori")
        self.apriori.run()
        #combine all attrsets
        self.attrsets = {}
        for f in self.apriori.freq:
            for a, d in f.items():
                self.attrsets[a] = d
        print(len(self.attrsets), "frequent attribute sets in data")
        if self.apriori.debug > 1: print(self.attrsets)
        self.apriori.computePositiveBorder()


    def compute_attrset_interestingness(self, free_vars):
        N = self.attrsets[()][()]
        edistr = self.joint.get_marginal(free_vars)
        # attribute sets interestingness
        # form array distribution from a dictionary
        dict = self.attrsets[tuple(free_vars)]
        #shape = [len(bn.attrdomains[i]) for i in free_vars]
        distr = numpy.zeros(edistr.shape)
        for v, n in dict.items():
            distr[v] = float(n)/N
        diff = numpy.array(numpy.fabs(edistr - distr))
        inter = max(numpy.ravel(diff))
        #inter = sum(numpy.ravel(diff))
        return inter, distr, edistr

    def run(self, minsup = 10, maxK = 5, apriori_debug = 1):
        self.compute_frequent_attr_sets_in_data(minsup, maxK, apriori_debug)
        
        self.thesop = create_sop(self.bn)
        self.attr_sets_w_inter = []
        N = self.attrsets[()][()]

        attrsetslist = self.attrsets.keys()

        # compute expected distributions
        self.joint = DiscreteDistrSOP(self.thesop)
        self.joint.get_positive_border_marginals(self.apriori.positiveBorder.keys())
        for free_vars in attrsetslist:
            edistr = self.joint.get_marginal(free_vars)

        # do Apriori on the BN
        dda = DDApriori(self.joint)
        dda.debug = 0
        dda.minsup = float(self.apriori.minsup) / N
        dda.maxK = self.apriori.maxK
        print("DDApriori, maxK =", dda.maxK)
        dda.run()
        #print(dda.freq)
        #find attr sets frequent in BN, whose distr in data is not known:
        attr_sets_to_count = {}
        for aset in itertools.chain(*dda.freq):
            if aset not in self.attrsets:
                attr_sets_to_count[aset] = {}
        # count supports of those attrsets
        self.ds.rewind()
        n = 0
        for rec in self.ds:
            n += 1
            for attrset, distr in attr_sets_to_count.items():
                vals = [rec[attrno] for attrno in attrset]
                distr[tuple(vals)] = distr.get(tuple(vals), 0) + 1
        self.attrsets.update(attr_sets_to_count)
        attrsetslist = self.attrsets.keys()

        # compute interestingness
        for free_vars in attrsetslist:
            inter, distr, edistr = self.compute_attrset_interestingness(free_vars)
            if inter > 0:
                self.attr_sets_w_inter.append((free_vars, inter))


        self.attr_sets_w_inter.sort(key = lambda x: -x[1])
        return self.attr_sets_w_inter

