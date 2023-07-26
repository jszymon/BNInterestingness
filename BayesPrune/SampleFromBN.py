#!/usr/bin/env python

"""Helper program for generating samples from a Bayesian network."""


import sys
import itertools
import random

import DataAccess.ArffFileReader
import Apriori.AprioriDistr
import BayesNet.BayesNet
import BayesNet.BayesHuginFile
import BayesNet.BayesNetApprox


#filename = "data/Munin1.hugin.gz"
filename = "data/Munin2.hugin.gz"
filename = "/home/szymon/dmining/data/Borreliosis/borrelia.net"
N = 30000
sep = '\t'
outfile = "/tmp/borelia.tabsep"
#outfile = "/mnt/data/Munin2.arff"
#outfile = "data/Munin1.arff"
#outfile = "data/Munin_final.arff"
#outfile = "-"
arff=False

random.seed(0)

if __name__ == "__main__":
    if outfile == "-":
        of = sys.stdout
    else:
        of = file(outfile, "w")
    bn = BayesNet.BayesHuginFile.read_Hugin_file(filename)
    print len(bn), "nodes"
    #print bn
    bn.normalizeProbabilities()
    bn.validate()

    # header
    if arff:
        print >>of, "@RELATION no-name"
        for a in bn:
            print >>of, "@ATTRIBUTE", a.attrname, "{", ",".join(a.attrdomain), "}"
        print >>of, "@DATA"
        sep = ","
    else:
        print >>of,sep.join(bn.attrnames)


    # data
    sampler = BayesNet.BayesNetApprox.BayesSampler(bn)
    for i in range(N):
        row = sampler.next()
        rowstr = ["'"+attr.attrdomain[x]+"'" for attr, x in itertools.izip(bn, row)]
        print >>of,sep.join(rowstr)
        if i % 1000 == 0:
            print "row", i
            sys.stdout.flush()
    of.close()
