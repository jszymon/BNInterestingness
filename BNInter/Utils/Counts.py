"""Compute counts for collections of attribute sets."""


import itertools
import sys
import time

import numpy as np

from .AttrSetCover import AttrSetCover

def dict_to_numpy(shape, distr):
    """Convert a dictionary counts for an attribute set to a numpy
    counts."""
    ndistr = np.zeros(shape)
    for values, P in distr.items():
        ndistr[values] = P
    return ndistr

def compute_counts_dict(asets, database, maxN = -1):
    """Compute counts for attribute sets in asets based on database.

    Returns a tuple with counts, number of rows in database and the
    number of missing values for each attribute set.  The coutns are
    returned as a dictionary of dictionaries.  The missing counts are
    returned as a dictionary containing number of missing values for
    each aset.  'None' in a database record is treated as missing
    value."""

    #initialization of helper arrays/dicts
    counts_helper = [(tuple(aset), {}) for aset in asets]
    missing_counts = {}
    for aset in asets:
        missing_counts[tuple(aset)] = 0
        
    if maxN >= 0:
        it = itertools.islice(iter(database), maxN)
    else:
        it = iter(database)
    N = 0
    for row in it:
        for aset, distr in counts_helper:
            vals = tuple([row[attrno] for attrno in aset])
            if None in vals:
                missing_counts[aset] = missing_counts[aset] + 1
            else:
                distr[vals] = distr.get(vals, 0) + 1
        N += 1
        if N % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print()
    counts = {}
    for aset, distr in counts_helper:
        counts[tuple(aset)] = distr
    return counts, N, missing_counts


def convert_counts_dicts2numpyarrays(domainsizes, counts):
    ncounts = {}
    for aset, distr in counts.items():
        shape = [domainsizes[attrno] for attrno in aset]
        ncounts[aset] = dict_to_numpy(shape, distr)
    return ncounts

def compute_counts_array(asets, domainsizes, database, maxN = -1):
    """Compute counts for attribute sets in asets based on database.

    Returns a tuple with counts and number of rows in database.
    The coutns are returned as a dictionary of numpy arrays."""
    counts, N, missing_counts = compute_counts_dict(asets, database, maxN)
    ncounts = convert_counts_dicts2numpyarrays(domainsizes, counts)
    return ncounts, N, missing_counts


def marginalize_numpy(a, vars, targetvars):
    """Marginalize a distribution over targetvars from
    distribution over vars.

    Deistributions are represented as numpy arrays."""
    vars = list(vars)
    axeslist = tuple(vars.index(x) for x in set(vars) - set(targetvars))
    a = np.sum(a, axis=axeslist)
    return a


class cost_functions(object):
    """Class containing functions returning costs used in set cover
    creation."""
    def __init__(self, domsizes, N):
        self.N = N
        self.domsizes = domsizes
    def query_cost(self, set):
        return self.N * len(set)
    def marginalization_cost(self, set, subset):
        domsize = 1
        for a in set:
            domsize *= self.domsizes[a]
        return len(subset)*min(self.N, domsize)
        #return len(subset)*min(self.N, 2**len(set)) # FIX: use correct attribute domain sizes, not 2
        #return (len(set)-len(subset)+1)*min(self.N, 2**len(set)) # FIX: use correct attribute domain sizes, not 2


def compute_counts_array_cover(asets, database, nattrs, domainsizes, maxN = -1):
    """Compute counts for attribute sets in asets based on database.

    Returns a tuple with counts, number of rows in database and the
    number of missing values for each attribute set.  The coutns are
    returned as a dictionary of numpy arrays.  The missing counts are
    returned as a dictionary containing number of missing values for
    each aset.  'None' in a database record is treated as missing
    value."""

    if not isinstance(database, list):
        print("Counts: iterator given, converting to list")
        if maxN >= 0:
            it = itertools.islice(iter(database), maxN)
        else:
            it = iter(database)
        data = list(it)
    else:
        data = database
    N = len(data)

    t = time.time()
    cover, covered, covers, gain = AttrSetCover(range(nattrs), asets,
                                                cost_functions(domainsizes, N).query_cost,
                                                cost_functions(domainsizes, N).marginalization_cost)
    print("cover gen. time =", time.time() - t)

    del cover, covered  # save some memory

    counts = {}
    missing_counts = {}
    for aset in asets:
        missing_counts[tuple(aset)] = 0
    
    print("Utils.Counts: total counts", len(asets))
    i = 0
    for c, covered_sets in covers.items():
        # compute counts for c - the covering set
        cover_distr = {}
        for row in data:
            vals = tuple([row[attrno] for attrno in c])
            cover_distr[vals] = cover_distr.get(vals, 0) + 1
        # marginalize subsets
        for aset in covered_sets:
            distr = {}
            indexmap = [list(c).index(attrno) for attrno in aset]
            for vals, cnt in cover_distr.items():
                vals2 = tuple([vals[attrno] for attrno in indexmap])
                #vals2 = tuple(vals[attrno] for attrno in indexmap)#2.4 only!!!
                if None in vals2:
                    missing_counts[aset] = missing_counts[aset] + cnt
                else:
                    distr[vals2] = distr.get(vals2, 0) + cnt
            shape = [domainsizes[x] for x in aset]
            counts[tuple(aset)] = dict_to_numpy(shape, distr)
            i += 1
            if i % 1000 == 0:
                sys.stdout.write("*")
                sys.stdout.flush()
    return counts, N, missing_counts


### TODO: this function should be called from previous one which
### should be very short
def compute_counts_dict_cover(asets, database, nattrs, domainsizes, maxN = -1):
    """Compute counts for attribute sets in asets based on database.

    Returns a tuple with counts, number of rows in database and the
    number of missing values for each attribute set.  The coutns are
    returned as a dictionary of dictionarys.  The missing counts are
    returned as a dictionary containing number of missing values for
    each aset.  'None' in a database record is treated as missing
    value."""

    if not isinstance(database, list):
        print("Counts: iterator given, converting to list")
        if maxN >= 0:
            it = itertools.islice(iter(database), maxN)
        else:
            it = iter(database)
        data = list(it)
    else:
        data = database
    N = len(data)

    t = time.time()
    cover, covered, covers, gain = AttrSetCover(range(nattrs), asets,
                                                cost_functions(domainsizes, N).query_cost,
                                                cost_functions(domainsizes, N).marginalization_cost)
    print("cover gen. time =", time.time() - t)

    del cover, covered  # save some memory

    counts = {}
    missing_counts = {}
    for aset in asets:
        missing_counts[tuple(aset)] = 0
    
    print("Utils.Counts: total counts", len(asets))
    i = 0
    for c, covered_sets in covers.items():
        # compute counts for c - the covering set
        cover_distr = {}
        for row in data:
            vals = tuple([row[attrno] for attrno in c])
            cover_distr[vals] = cover_distr.get(vals, 0) + 1
        # marginalize subsets
        for aset in covered_sets:
            distr = {}
            indexmap = [list(c).index(attrno) for attrno in aset]
            for vals, cnt in cover_distr.items():
                vals2 = tuple([vals[attrno] for attrno in indexmap])
                #vals2 = tuple(vals[attrno] for attrno in indexmap)#2.4 only!!!
                if None in vals2:
                    missing_counts[aset] = missing_counts[aset] + 1
                else:
                    distr[vals2] = distr.get(vals2, 0) + cnt
            counts[tuple(aset)] = distr
            i += 1
            if i % 1000 == 0:
                sys.stdout.write("*")
                sys.stdout.flush()
    return counts, N, missing_counts




#def compute_counts_dict_cover(asets, database, nattrs, maxN = -1,
#                              query_cost_function = query_cost,
#                              marginalization_cost_function = marginalization_cost):
#    """Compute counts for attribute sets in asets based on database.
#
#    Returns a tuple with counts and number of rows in database.
#    The coutns are returned as a dictionary of dictionaries."""
#    cover, covered, covers, gain = AttrSetCover(range(nattrs), asets, query_cost_function,
#                                                     marginalization_cost_function)
#    #cover_distributions, N = compute_counts_dict(cover, database, maxN)
#
#
#    if maxN >= 0:
#        it = itertools.islice(iter(database), maxN)
#    else:
#        it = iter(database)
#    data = list(it)
#
#
#    counts = {}
#    for c, covered_sets in covers.iteritems():
#        # compute counts for c
#        cover_distr = {}
#        for row in it:
#            vals = tuple([row[attrno] for attrno in c])
#            cover_distr[vals] = cover_distr.get(vals, 0) + 1
#        # marginalize subsets
#        for aset in covered_sets:
#            distr = {}
#            indexmap = [c.index(attrno) for attrno in aset]
#            for vals, cnt in cover_distr.iteritems():
#                vals2 = tuple([vals[attrno] for attrno in indexmap])
#                distr[vals2] = distr.get(vals2, 0) + cnt
#            counts[tuple(aset)] = distr
#        
#
#    # marginalize from cover
#    #counts = {}
#    #print "Utils.Counts: total counts", len(asets)
#    #i = 0
#    #for aset in asets:
#    #    distr = {}
#    #    c = covered[tuple(aset)]
#    #    indexmap = [c.index(attrno) for attrno in aset]
#    #    for vals, cnt in cover_distributions[tuple(c)].iteritems():
#    #        vals2 = tuple([vals[attrno] for attrno in indexmap])
#    #        distr[vals2] = distr.get(vals2, 0) + cnt
#    #    counts[tuple(aset)] = distr
#    #    i += 1
#    #    if i % 1000 == 0:
#    #        sys.stdout.write("*")
#    #        sys.stdout.flush()
#
#    return counts, N


def test_counts(asets, data):
    counts, N, missing_counts = compute_counts_dict(asets, data)
    print(N)
    print(counts)
    print(missing_counts)

    print(dict_to_numpy((2,2), counts[(0,1)]))

    counts, N, missing_counts = compute_counts_array(asets, [2,2,2], data)
    print(N)
    print(counts)
    print(missing_counts)

    #counts, N = compute_counts_dict_cover(asets, data, 3)
    #print N
    #print counts
    #print missing_counts

    counts, N, missing_counts = compute_counts_array_cover(asets, data, 3, [2,2,2])
    print(N)
    print(counts)
    print(missing_counts)
    


if __name__ == "__main__":
    print("small data, no missing values")
    asets = [[0,1], [0,2], [1,2]]
    data = [[0,0,1],[0,0,1],[1,1,1]]
    test_counts(asets, data)

    print()
    print("small data, with missing values")
    asets = [[0,1], [0,2], [1,2]]
    data = [[0,None,1],[0,0,1],[1,None,None]]
    test_counts(asets, data)
