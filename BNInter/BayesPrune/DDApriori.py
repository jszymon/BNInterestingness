"""Mining frequent patterns from a discrete distribution."""

from ..Apriori.AprioriDistr import AprioriDistr


class DDApriori(AprioriDistr):
    def __init__(self, discr_distr):
        self.discr_distr = discr_distr
        self.maxK = 10
        self.minsup = 0.1
        self.debug = 0
        self.__negativeBorder = {}
    def init_cand(self):
        self.cand = {}
        for i in range(self.discr_distr.sop.n):
            self.cand[(i,)] = None
    def count_support(self):
        self.discr_distr.get_positive_border_marginals(self.cand.keys())
        for attrset in self.cand:
            self.cand[attrset] = self.discr_distr.get_marginal(attrset)
        return 1.0
    def find_frequent(self):
        self.freq.append({})
        for attrset, distr in self.cand.items():
            # use MAX for now, could use entropy etc.
            supp = distr.max()
            if supp >= self.minsup:
                self.freq[-1][attrset] = distr
            else:
                self.__negativeBorder[attrset] = distr

