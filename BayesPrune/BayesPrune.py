#!/usr/bin/env python


import numpy
import itertools
import sets
import random
import sys

sys.path.append("..")

from DataAccess import create_arff_reader
from BayesNet import BayesNet, read_Hugin_file, write_Hugin_file
import BayesNet.BayesNetLearn
import Apriori.AprioriDistr
import time
from ExactInterestingness import BN_interestingness_exact
from SamplingInterestingness import BN_interestingness_sample





global debug
debug = 0




global minsup
global maxK
minsup = 10
maxK = 5
#minsup = 50
#maxK = 3



def read_bn_network(base_name):
    """Read a Bayesian network from a Hugin file."""
    global debug
    filename = base_name + ".net"
    bn = read_Hugin_file(filename)
    if debug > 1:
        print bn
    elif debug > 0:
        print len(bn), "nodes in BN"
    return bn



def print_attr_sets_with_inter(attr_sets_w_inter, maxlen = 10000, maxN = 10, must_contain_attrno = None,
                               mode = ["attrset"], bn_interestingness = None):
    asets_selected = [x for x in attr_sets_w_inter if len(x[0]) <= maxlen]
    if must_contain_attrno != None:
        asets_selected = [x for x in asets_selected if must_contain_attrno in x[0]]
    for aset, inter in [(a,i) for a, i in asets_selected][:maxN]:
        if "attrset" in mode:
            print "[" + ",".join([bn_interestingness.ds.attrset[i].name for i in aset]) + "] " + str(inter)
        if "maxcell" in mode:
            inter, distr, edistr = bn_interestingness.compute_attrset_interestingness(aset)
            diff = numpy.abs(distr - edistr)
            indices = numpy.ndindex(diff.shape)
            for i, d in itertools.izip(indices, numpy.ravel(diff)):
                if d >= 0.9 * inter:
                    idomlist = [str(bn_interestingness.ds.attrset[a].domain[v]) for a, v in itertools.izip(aset, i)]
                    istr = ",".join(idomlist)
                    #print i, distr, edistr
                    print (istr + " => " + str(d) + "  P^BN=" + str(edistr[i]) +"  P^D=" + str(distr[i]))

def topoPrune(bn, attr_sets_w_inter, mininter):
    # keep only interesting attrsets
    asets = [x for x in attr_sets_w_inter if x[1] >= mininter]
    import Utils.SEtree
    setree = Utils.SEtree.SEtree()
    for aset, inter in asets:
        setree[aset] = inter
    import BayesNet.BayesNetGraph
    pruned = []
    for aset, inter in asets:
        asetnames = bn.attrNumbers2Names(aset)
        ancestors = BayesNet.BayesNetGraph.ancestors(bn, asetnames)
        ancestors.update(asetnames)
        ancestorsnumbers = bn.attrNames2Numbers(ancestors)
        #print asetnames, ancestors
        l = list(setree.iter_included(ancestorsnumbers))
        l = [a for a in l if not sets.Set(a) ==  sets.Set(aset)] # this does topo + hierarchical
        #l = [a for a in l if not sets.Set(a).issuperset(sets.Set(aset))] # this does topo + hierarchical
        #l = [a for a in l if not sets.Set(a).issuperset(sets.Set(aset)) and not sets.Set(a).issubset(sets.Set(aset))] # just topo
        if len(l) == 0:
            pruned.append((aset, inter))
    return pruned
#--------------------------------

def testRun(bn):
    """Run a standard test."""
    debug = 1
    t1 = time.clock()

    ### Find interesting patterns

    # Exact computations
    bn_interestingness = BN_interestingness_exact(bn, ds)
    attr_sets_w_inter = bn_interestingness.run(minsup = minsup, maxK = maxK, apriori_debug = debug)
    # Sampling
    #bn_interestingness = BN_interestingness_sample(bn, ds)
    #attr_sets_w_inter = bn_interestingness.run(maxK = maxK, excluded_attrs = excluded_attrs, selectionCond = selectionCond)



    ### print results
    t2 = time.clock()
    print "Inter time=" + str(t2-t1)
    #print_attr_sets_with_inter(attr_sets_w_inter, 1000, len(attr_sets_w_inter), mode = ["attrset", "maxcell"], bn_interestingness = bn_interestingness)
    # smaller results:
    print_attr_sets_with_inter(attr_sets_w_inter, 1000, len(attr_sets_w_inter), mode = ["attrset"], bn_interestingness = bn_interestingness)

    #print "\nTopological pruning:\n"
    #attr_sets_w_inter = topoPrune(bn, attr_sets_w_inter, 0.01)
    #print_attr_sets_with_inter(attr_sets_w_inter, 1000, len(attr_sets_w_inter))
    

def buildNetworkInteractively(bn, data_scanner):
    quit = False
    data = list(ds)
    must_contain_attr = None # attribute that displayed attrsets must contain
    nattrsets = 10
    maxlen = 1000
    while quit != True:
        bn.validate()
        ds.rewind()
        BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)
        print "Interestingness with respect to"
        print bn
        print BayesNet.BayesNetLearn.lnP_dataset_cond_network_structure(bn, data)
        t1 = time.clock()
        bn_interestingness = BN_interestingness_exact(bn, ds)
        attr_sets_w_inter = bn_interestingness.run(minsup = minsup, maxK = maxK, apriori_debug = debug)
        #bn_interestingness = BN_interestingness_sample(bn, ds)
        #attr_sets_w_inter = bn_interestingness.run(maxK = maxK)
        t2 = time.clock()
        print "Inter time=" + str(t2-t1)


        #attr_sets_w_inter = topoPrune(bn, attr_sets_w_inter, 0.01)


        while True:
            attrnames = [n.name for n in bn]
            if must_contain_attr != None:
                must_contain_no = bn.names_to_numbers([must_contain_attr])[0]
            else:
                must_contain_no = None
            print_attr_sets_with_inter(attr_sets_w_inter, maxlen, nattrsets, must_contain_attrno = must_contain_no,
                                       mode = ["attrset", "maxcell"], bn_interestingness = bn_interestingness)
            print
            print "1. add edge"
            print "2. delete edge"
            print "3. limit attrset size"
            print "4. remove all edges"
            print "5. print network"
            print "6. limit number of displayed sets"
            print "7. set 'must have' attribute"
            print "8. save network"
            print "Q. quit"
            choice = raw_input("-->")
            if choice in ["q", "Q"]:
                quit = True
                break
            elif choice == "3":
                maxlen = int(raw_input("Enter max length: "))
            elif choice == "1":
                src = raw_input("Enter from attribute: ")
                if src not in attrnames:
                    print "wrong attribute name"
                    continue
                dst = raw_input("Enter to attribute: ")
                if dst not in attrnames:
                    print "wrong attribute name"
                    continue
                bn.addEdge(src, dst)
                break
            elif choice == "2":
                src = raw_input("Enter from attribute: ")
                dst = raw_input("Enter to attribute: ")
                bn.delEdge(src, dst)
                break
            elif choice == "4":
                BayesNet.BayesNetLearn.makeIndependentStructure(bn)
                break
            elif choice == "5":
                print bn
            elif choice == "6":
                try:
                    nattrsets = int(raw_input("Enter number of attrsets shown: "))
                except ValueError:
                    print "not an integer"
            elif choice == "7":
                must_contain_attr = raw_input("Enter the 'must contain' attribute: ").strip()
                if must_contain_attr == "":
                    must_contain_attr = None
                if must_contain_attr not in attrnames:
                    print "wrong attribute name"
                    must_contain_attr = None
                    continue
            elif choice == "8":
                fname = raw_input("Enter file name: ")
                of = file(fname, "wU")
                write_Hugin_file(bn, of)
                of.close()
                continue
            else:
                print "Wrong choice"
    

if __name__ == "__main__":
    excluded_attrs = []
    selectionCond = None
    base_name = "data/ksl_discr"  # suffixes will be added here
    minsup = 10
    maxK = 5
    #base_name = "data/ksl_discr_subset"  # data attrs are subset of network attrs
    #minsup = 10
    #maxK = 4
    #base_name = "data/ksl_discr_missing"  # suffixes will be added here
    #minsup = 10
    #maxK = 4
    base_name = "data/soybean"  # suffixes will be added here
    minsup = 50
    maxK = 3
    #base_name = "data/breast-cancer"  # suffixes will be added here
    #minsup = 2
    #maxK = 5
    #base_name = "data/anneal_discr3"  # suffixes will be added here
    #minsup = 9
    #maxK = 3
    #base_name = "data/mushroom"  # suffixes will be added here
    #minsup = 81
    #maxK = 3
    #base_name = "data/audiology"  # suffixes will be added here
    #minsup = 2
    #maxK = 3
    #base_name = "data/lymph"  # suffixes will be added here
    #minsup = 1
    #maxK = 3
    #base_name = "data/splice"  # suffixes will be added here
    #minsup = 31
    #maxK = 4
    #base_name = "/home/szymon/dmining/data/DrCiechanowicz/dSS/dSS_dscr_train"  # suffixes will be added here
    #minsup = 1
    #maxK = 4
    #base_name = "/home/szymon/dmining/data/DrCiechanowicz/stulatkowie/stulatkowie"  # suffixes will be added here
    #minsup = 1
    #maxK = 4
    #base_name = "data/Munin_final_150"  # suffixes will be added here
    #minsup = 1
    #maxK = 3
    #base_name = "data/Munin2_final_100"  # suffixes will be added here
    #minsup = 1
    #maxK = 2
    #base_name = "/home/szymon/dmining/data/Borreliosis/borrelia"  # suffixes will be added here
    #minsup = 1
    #maxK = 10
    #excluded_attrs = ['Insectbite','Duration','Month']
    #def select(row): return row[77] == 0
    #selectionCond = select
   

    if len(sys.argv) >= 2:
        random.seed(int(sys.argv[1]))
    
    debug = 1


    filename = base_name + ".arff"
    ds = create_arff_reader(filename)
    ds.rewind()

    BNread = True
    try:
        bn = read_bn_network(base_name)
    except BayesNet.BayesHuginFile.BNReadError, bne:
        print str(bne)
        BNread = False
    if BNread == False:
        print "assuming independent structure"
        bn = BayesNet.BayesNet.BayesNet(ds.filename, [a.name for a in ds.attrset], [a.domain for a in ds.attrset])
        BayesNet.BayesNetLearn.makeIndependentStructure(bn)
        ds.rewind()
        BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)

        
    #print BayesNet.BayesNetLearn.lnP_dataset_cond_network_structure(bn, ds)
    ds.rewind()

    #BayesNet.BayesNetLearn.makeIndependentStructure(bn)
    #ds.rewind()
    #BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)


    #bn.addEdge("sex","casecon")
    #bn.addEdge("ace","anp34")
    #bn.addEdge("ace","ampd1")
    #bn.addEdge("ampd1", "anp12")
    #bn.addEdge("ampd1", "casecon")
    #print bn
    #ds.rewind()
    #BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)


    # use independence structure
    #BayesNet.BayesNetLearn.makeIndependentStructure(bn)
    #bn.addEdge("KAL4","KAL5")
    #bn.addEdge("PROK","KAL1")
    #bn.addEdge("wiek","MAP")
    #bn.addEdge("PROK","KAL4")
    #bn.addEdge("PROK","KAL5")
    #bn.addEdge("PROK","PROK_A")
    #bn.addEdge("PROK","PROK_B")
    #bn.addEdge("PROK","PROK_H")
    #bn.addEdge("PROK","PROK_I")
    #bn.addEdge("PROK","PROK_Q")
    #bn.addEdge("PROK","PROK_R")
    #bn.addEdge("BE16","BE27")
    #bn.addEdge("wiek","BMI")
    ##bn.addEdge("plec","dSS")
    ##bn.addEdge("ACE","dSS")
    #print bn
    #ds.rewind()
    #BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)

    #common sense
    #BayesNet.BayesNetLearn.makeIndependentStructure(bn)
    #bn.addEdge("Sex", "Smok")
    #bn.addEdge("Sex", "BMI")
    #bn.addEdge("Sex", "Work")
    #bn.addEdge("Sex", "Alc")
    #bn.addEdge("BMI", "Hyp")
    #bn.addEdge("Smok", "Hyp")
    #bn.addEdge("Smok", "FEV")
    #bn.addEdge("Year", "Work")
    #bn.addEdge("Year", "BMI")
    #bn.addEdge("Alc", "Hyp")
    #ds.rewind()
    #BayesNet.BayesNetLearn.learnProbabilitiesFromData(bn, ds, priorN = 0)
    #
    #bn.addEdge("Sex", "FEV")
    #bn.addEdge("Year", "Alc")
    #bn.addEdge("Year", "Kol")
    #bn.addEdge("Sex", "Kol")



    #bn.addEdge("Year", "BMI")
    #bn.addEdge("Year", "BMI")
    #bn.addEdge("Year", "FEV")
    #bn.addEdge("Alc", "FEV")

    
    #interestingness aided build
    #bn.addEdge("Sex", "FEV")
    #bn.addEdge("Sex", "Smok")
    #bn.addEdge("Year", "Alc")
    #bn.addEdge("Sex", "Alc")
    #bn.addEdge("Year", "Kol")
    #bn.addEdge("Sex", "Kol")
    #bn.addEdge("Sex", "BMI")
    #bn.addEdge("BMI", "Hyp")
    #bn.addEdge("Year", "Work")
    #bn.addEdge("Sex", "Work")
    #bn.addEdge("Year", "Smok")
    #bn.addEdge("Smok", "Hyp")
    
    #buildNetworkInteractively(bn, ds)
    testRun(bn)

    #print "Bayes prune"
    # group itemsets by attribute set
    #groupAndPrintsAttrsets(isets_w_inter, attrnames, domains)



