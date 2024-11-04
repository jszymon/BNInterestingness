#!/usr/bin/env python

"""Helper program for generating samples from a Bayesian network."""


import sys
import itertools
import random

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
        of = open(outfile, "w")
    bn = BayesNet.BayesHuginFile.read_Hugin_file(filename)
    print(len(bn), "nodes")
    #print(bn)
    bn.normalizeProbabilities()
    bn.validate()

    # header
    if arff:
        print("@RELATION no-name", file=of)
        for a in bn:
            print("@ATTRIBUTE", a.attrname, "{", ",".join(a.attrdomain), "}", file=of)
        print("@DATA", file=of)
        sep = ","
    else:
        print(sep.join(bn.attrnames), file=of)


    # data
    sampler = BayesNet.BayesNetApprox.BayesSampler(bn)
    for i in range(N):
        row = sampler.next()
        rowstr = ["'"+attr.attrdomain[x]+"'" for attr, x in itertools.izip(bn, row)]
        print(sep.join(rowstr), file=of)
        if i % 1000 == 0:
            print("row", i)
            sys.stdout.flush()
    of.close()
