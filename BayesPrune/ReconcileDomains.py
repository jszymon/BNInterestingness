#!/usr/bin/env python

"""Helper program for reconciling domains of Bayesian networks.

It is useful if some datavalues are absent and so B-course does not
include those vaues in domains.

The input are two networks, the output network has the domain of one
of them and connections of another.  Probabilities are learned from
the specified data file."""


import sys
import itertools
import copy

import DataAccess.ArffFileReader
import Apriori.AprioriDistr
import BayesNet.BayesNet
import BayesNet.BayesNetLearn
import BayesNet.BayesHuginFile
import BayesNet.BayesNetApprox

domainsnetwork = "data/Munin2.hugin.gz"  # network to take the domains from
connectionsnetwork = "data/Munin2_100.net"   # network to take connections from
datafile = "data/Munin2_100.arff"  # data to learn probabilities from
N = 100 # number of rows to learn probabilities on
outfile = "-"
outfile = "data/Munin2_final_100.net"

if __name__ == "__main__":
    if outfile == "-":
        of = sys.stdout
    else:
        of = file(outfile, "w")
    dombn = BayesNet.BayesHuginFile.read_Hugin_file(domainsnetwork)
    conbn = BayesNet.BayesHuginFile.read_Hugin_file(connectionsnetwork)
    # target network with domain from dombn and connections from conbn
    outbn = copy.deepcopy(dombn)

    BayesNet.BayesNetLearn.makeIndependentStructure(outbn)
    for node in conbn:
        name1 = node.attrname
        for name2 in node.parentnames:
            outbn.addEdge(name2, name1)
    
    ar = DataAccess.ArffFileReader.create_arff_reader(datafile)
    it = itertools.islice(iter(ar), N)
    BayesNet.BayesNetLearn.learnProbabilitiesFromData(outbn, it, priorN = 1)
    BayesNet.BayesHuginFile.write_Hugin_file(outbn, of)
    
