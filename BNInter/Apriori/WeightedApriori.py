"""Apriori allowing weights for rows"""


import Apriori


class WeightedApriori(Apriori.Apriori):
    def __init__(self, itemset_reader, W):
        super(WeightedApriori, self).__init__(itemset_reader)
        self.W = W

    def count_support(self):
        self.ir.rewind()
        N = 0
        for iset in self.ir:
            for x in self.cand.iter_included(iset):
                #x.item += 1#breaks encapsulation for speed
                self.cand[x] += self.W[N]
            N += 1
        return N

