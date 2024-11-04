import itertools

from .SEtree import SEtree

def compute_gain(s, setree, included_sets, query_cost_function, marginalization_cost_function):
    gain = -query_cost_function(s)
    for sub_s in itertools.chain(setree.iter_included(tuple(s)), included_sets):
        gain = gain + query_cost_function(sub_s) - marginalization_cost_function(s, sub_s)
    return gain

def __remove_included(setree, best_s):
    sets_to_remove = [s for s in setree.iter_included(best_s)]
    for s in sets_to_remove:
        del setree[s]
    return sets_to_remove

def AttrSetCover(U, sets, query_cost_function, marginalization_cost_function, method="1"):
    """Forms a collection of sets covering all sets in sets.

    Works by picking a set and generating its supersets until it
    causes a gain in a cost function.  Inspired bu Ullman's algorithm
    for materializing views.

    It might be beneficial to pass just the positive border of sets.

    U - possible set elements (usually of database attributes)
    sets -
    query_cost_function -
    marginalization_cost_function -"""
    cover = []
    covered = {}
    covers = {}
    tot_gain = 0
    #print("creating SEtree")
    setree = SEtree()
    for s in sets:
        setree[s] = None
    while len(setree) > 0:
        seed = setree.popitem()[0]
        setree[seed] = None
        best_s = list(seed)
        included_sets = []
        best_gain = compute_gain(seed, setree, included_sets, query_cost_function, marginalization_cost_function)
        nextiter = True
        while nextiter:
            #print("size=", len(best_s))
            included_sets.extend(__remove_included(setree, best_s))
            nextiter = False

            elem_gains = {}
            for e in U:
                if e in best_s:
                    continue
                elem_gain = -query_cost_function(best_s + [e])
                for s in included_sets:
                    elem_gain += query_cost_function(s) - marginalization_cost_function(best_s + [e], s)
                elem_gains[e] = elem_gain
            for set, e in setree.iter_one_not_in(best_s):
                #print(best_s, set,)
                if e in best_s:
                    print("!!!!ERROR!!!!")
                elem_gains[e] += query_cost_function(set) - marginalization_cost_function(best_s + [e], set)

            for e, gain in elem_gains.items():
                if gain > best_gain:
                    best_gain = gain
                    best_s = best_s + [e]
                    best_s.sort()
                    nextiter = True
        #print(best_s, best_gain)
        cover.append(best_s)
        tot_gain += best_gain
        included_sets.extend(__remove_included(setree, best_s))
        covered[seed] = best_s
        for s in included_sets:
            covered[s] = best_s
        covers[tuple(best_s)] = included_sets
    return cover, covered, covers, tot_gain


#def AttrSetCover(U, sets, query_cost_function, marginalization_cost_function, method="1"):
#    """Forms a collection of sets covering all sets in sets.
#
#    Works by picking a set and generating its supersets until it
#    causes a gain in a cost function.  Inspired bu Ullman's algorithm
#    for materializing views.
#
#    It might be beneficial to pass just the positive border of sets.
#
#    U - possible set elements (usually of database attributes)
#    sets -
#    query_cost_function -
#    marginalization_cost_function -"""
#    cover = []
#    covered = {}
#    covers = {}
#    tot_gain = 0
#    #print("creating SEtree")
#    setree = SEtree()
#    for s in sets:
#        setree[s] = None
#    while len(setree) > 0:
#        seed = setree.popitem()[0]
#        setree[seed] = None
#        best_s = list(seed)
#        included_sets = []
#        best_gain = compute_gain(seed, setree, included_sets, query_cost_function, marginalization_cost_function)
#        nextiter = True
#        while nextiter:
#            #print("size=", len(best_s))
#            included_sets.extend(__remove_included(setree, best_s))
#            nextiter = False
#            cur_s = best_s
#            for e in U:
#                #print "doing: best + ", e
#                if e in cur_s: continue
#                tmp_s = cur_s + [e]
#                tmp_s.sort()
#                gain = compute_gain(tmp_s, setree, included_sets, query_cost_function, marginalization_cost_function)
#                if gain > best_gain:
#                    best_gain = gain
#                    best_s = tmp_s
#                    nextiter = True
#        #print best_s, best_gain
#        cover.append(best_s)
#        tot_gain += best_gain
#        included_sets.extend(__remove_included(setree, best_s))
#        covered[seed] = best_s
#        for s in included_sets:
#            covered[s] = best_s
#        covers[tuple(best_s)] = included_sets
#    return cover, covered, covers, tot_gain


def query_cost(set):
    N = 1000
    return N*len(set)

def marginalization_cost(set, subset):
    return 2**len(set)

if __name__ == "__main__":
    sets = []
    n = 30
    for x in range(n):
        for y in range(x+1,n):
            for z in range(y+1,n):
                sets.append((x,y,z))
    print(sets, len(sets))
    cov, covered, covers, gain = AttrSetCover(range(n), sets, query_cost, marginalization_cost)
    print(cov, len(cov), gain, sum([query_cost(s) for s in sets]))
    print(covered)
