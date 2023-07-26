from .Apriori import Apriori
import Utils.SEtree
import Utils.Counts
import DataAccess.ArffFileReader
import numpy
import itertools


class AprioriDistr(Apriori):
    
    """The Apriori algorithm framework implementation where for every
    attribute set its whole distribution is stored."""

    def __init__(self, arff_scanner):
        self.db = arff_scanner
        self.maxK = 3
        self.minsup = 2
        self.debug = 0
        self.__negativeBorder = {}
    def init_cand(self):
        self.cand = {}
        attrs = self.db.get_attr_set()
        for i in range(len(attrs)):
            #self.cand[(i,)] = numpy.zeros(as[i].get_n_categories())
            self.cand[(i,)] = {}
    def add_empty_itemset(self, N):
        """Adds an empty attributeset with distribution (N)."""
        self.freq.append({})
        self.freq[-1][()] = {():N}
        #self.freq[-1][()] = numpy.array(float(N))
    def count_support2(self):
        self.db.rewind()
        N = 0
        for rec in self.db:
            N += 1
            for attrset, distr in self.cand.items():
                vals = [rec[attrno] for attrno in attrset]
                distr[tuple(vals)] = distr.get(tuple(vals), 0) + 1
        return N

    def count_support(self):
        self.db.rewind()
        domsizes = [len(a.domain) for a in self.db.get_attr_set()]
        counts, N, missing_counts = Utils.Counts.compute_counts_dict_cover(self.cand.keys(), self.db, len(self.db.get_attr_set()), domsizes)
        for aset in self.cand:
            self.cand[aset] = counts[aset]
        return N

    def find_frequent(self):
        self.freq.append({})
        for attrset, distr in self.cand.items():
            # use MAX for now, could use entropy etc.
            supp = max(distr.values())
            if supp >= self.minsup:
                self.freq[-1][attrset] = distr
            else:
                self.__negativeBorder[attrset] = distr
    def make_candidates(self):
        self.cand.clear()
        tmp = list(self.freq[-1].keys())
        tmp.sort()
        it = iter(tmp)
        first = next(it)
        joinable = [first]
        for iset in it:
            if first[0:self.K-1] == iset[0:self.K-1]:
                joinable.append(iset)
            else:
                self.__join_itemsets(joinable)
                first = iset
                joinable = [iset]
        self.__join_itemsets(joinable)
    def __join_itemsets(self, joinable):
        for x in range(len(joinable)):
            left = joinable[x]
            for right in joinable[x+1:]:
                attr_set = left + (right[-1],)
                self.cand[attr_set] = {}
    def computePositiveBorder(self):
        """Computes the positive border."""
        self.positiveBorder = {}
        # all largest itemsets are in Bd+
        # treating separately is useful when maxK is used
        self.positiveBorder = self.freq[-1].copy()
        # all direct subsets of sets in neg. border are in pos. border
        for aset in self.__negativeBorder:
            for i in range(len(aset)):
                subset = aset[0:i]+aset[i+1:]
                if not subset in self.positiveBorder:
                    self.positiveBorder[subset] = self.freq[len(subset)][subset]




if __name__ == '__main__':
    file = "../../../data/lenses.arff"
    #file = "../../../data/arff/agaricus-lepiota.arff"
    #file = "../ksl_discr_year_2.arff"
    file = "../ksl_discr.arff"
    ds = DataAccess.ArffFileReader.create_arff_reader(file)

    import time
    t1 = time.clock()
    
    ap = AprioriDistr(ds)
    ap.minsup = 10
    ap.maxK = 50
    ap.debug = 1
    print("Apriori")
    ap.run()

    t2 = time.clock()
    print("Apriori time=" + str(t2-t1))
    for f in ap.freq:
        print(f)


    #print itemsets in readable format
    # outf = open("itemsets", "w")
    # print >>outf,repr(ris.itemset_descr)
    # print >>outf,ap.N
    # isets = []
    # for fset in ap.freq:
    #     for iset in fset:
    #         ilist = [(ris.itemset_descr[i][0], ris.itemset_descr[i][2]) for i in iset]
    #         isets.append((ilist, fset[iset]))
    # print >>outf,repr(isets)
    # outf.close()

    # print
    # rb = RuleBase.RuleBase()
    # rb.add_from_itemsets_with_consequent(ap.freq, 93)
    # print rb
    # 
    # r = rb.rules[2].itervalues().next()
    # print "rule",r
    # print "subrules: ",rb.subrules(r)

