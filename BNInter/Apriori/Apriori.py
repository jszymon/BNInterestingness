from ..Utils.SEtree import SEtree


class Apriori:
    """The Apriori algorithm framework"""
    def __init__(self, itemset_reader):
        self.ir = itemset_reader
        self.maxK = 3
        self.minsup = 2
        self.debug = 0

    def initialize(self):
        """Initialize the algorithm."""
        self.init_cand()
        self.freq = []
        self.K = 1
        
    def run(self):
        self.initialize()
        while len(self.cand) > 0 and self.K <= self.maxK:
            if self.debug > 0:
                print("K=",self.K)
            if self.debug > 1:
                print(self.cand)
            if self.debug > 0:
                print("K=",self.K,"counting support")
            N = self.count_support()
            if self.K == 1:
                self.add_empty_itemset(N)
            if self.debug > 1:
                print(self.cand)
            self.find_frequent()
            if self.debug > 1:
                print(self.cand)
            if len(self.freq[-1]) == 0:
                break
            if self.K < self.maxK:
                if self.debug > 0:
                    print("K=",self.K,"generating candidates")
                self.make_candidates()
                self.prune_candidates()
            self.K += 1
        self.N = N
    def init_cand(self):
        self.cand = SEtree()
        for i in range(self.ir.n_items):
            self.cand[[i]] = 0
    def add_empty_itemset(self, N):
        """Adds an empty itemset with support N."""
        self.freq.append(SEtree())
        self.freq[-1][[]] = N
    def count_support(self):
        self.ir.rewind()
        N = 0
        for iset in self.ir:
            N += 1
            for x in self.cand.iter_included(iset):
                #x.item += 1#breaks encapsulation for speed
                self.cand[x]+=1
        return N
    def find_frequent(self):
        self.freq.append(Utils.SEtree.SEtree())
        for iset, cnt in self.cand.iteritems():
            if cnt >= self.minsup:
                self.freq[-1][iset] = cnt
        #print "freq",self.K
        #print self.freq[-1]
    def make_candidates(self):
        self.cand.clear()
        it = iter(self.freq[-1])
        first = it.next()
        joinable = [first]
        for iset in it:
            if first[0:self.K-1] == iset[0:self.K-1]:
                joinable.append(iset)
            else:
                self.__join_itemsets(joinable)
                first = iset
                joinable = [iset]
        self.__join_itemsets(joinable)
    def prune_candidates(self):
        """remove candidates with infrequent subsets"""
        for c in self.cand.keys():
            for i in range(len(c) - 2):
                cc = c[0:i]+c[i+1:]
                if cc not in  self.freq[-1]:
                    del self.cand[c]
                    break
    def __join_itemsets(self, joinable):
        for x in range(len(joinable)):
            left = joinable[x]
            for right in joinable[x+1:]:
                self.cand[left + (right[-1],)] = 0


# file = "../../../data/lenses.arff"
# #file = "../../../data/arff/agaricus-lepiota.arff"
# file = "../ksl_discr_year_2.arff"
# ris = record_to_itemset_scanner.record_to_itemset_scanner(ds)
# 
# ap = Apriori(ris)
# ap.minsup = 10
# ap.maxK = 50
# print "Apriori"
# ap.run()
# 
# for f in ap.freq:
#     print f
# 
# 
# #print itemsets in readable format
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

