import math
import itertools
import tempfile
import os

import numpy as np

from ..Utils.gaussinv import cdf_ugaussian_Pinv
from ..Utils.Counts import compute_counts_array_cover
from ..DataAccess import ProjectionReader
from ..DataAccess import SelectionReader
from ..BayesNet import BayesSampler



def E_Chernoff(m, delta):
    """Interval width based on Chernoff bounds for given sample size
    m, and confidence 1-delta."""
    return math.sqrt(math.log(2.0 / delta) / (2 * m))

def N_Chernoff(E, delta):
    """Number of samples required to get Chernoff bounds narrower or
    eq. E for confidence 1-delta."""
    return int(math.ceil(math.log(2.0 / delta) / (2 * E * E)))

def M_Chernoff(epsilon, nhyp, delta):
    """Worst case sample size for interval size epsilon, number
    of hypotheses nhyp, confidence 1-delta."""
    return 2.0/(epsilon * epsilon)*math.log(4 * nhyp / delta)





### TWO-WAY SAMPLING


#def f(i, k = 15):
#    return 1.0 / ((i+k)*(i+k)) / ((math.pi*math.pi) / 6.0 - sum([1.0/((x+1)*(x+1)) for x in range(k-1)]))
def f(i):
    return 6.0 / (math.pi*math.pi) / (i*i)
#def f(i):   # that's the theoretically best possible bound we could get...
#    return 1

class small_attr_set(object):
    """Small version of attribute set class used to save memory when
    full version is not needed.

    __slots__ are used to conserve memory."""
    __slots__ = ['key', 'inter', 'inter_ci', 'supp', 'supp_ci', 'offset']
    def __init__(self, attr_set):
        self.key = attr_set.key
        self.inter = attr_set.inter
        self.inter_ci = attr_set.inter_ci
        self.supp = attr_set.supp
        self.supp_ci = attr_set.supp_ci
    def fromfile(self, file):
        """Read and return the corresponding full attribute set from
        file"""
        fullh = attr_set(self.key)
        fullh.fromfile(file)
        fullh.inter = self.inter
        fullh.inter_ci = self.inter_ci
        fullh.supp = self.supp
        fullh.supp_ci = self.supp_ci
        return fullh

class attr_set(object):
    sample_from_data = False
    ND = 0
    domainsizes = None

    def __update_fields(self):
        """compute values for various fields"""
        self.shape = [attr_set.domainsizes[a] for a in self.key]
        self.domsize = 1
        for x in self.shape:
            self.domsize *= x
        

    def __init__(self, aset):
        self.key = aset
        self.__update_fields()

        self.N_data = 0 #how many samples used to count from data
        self.counts_data = np.zeros(self.shape)
        self.N_model = 0 #how many samples used to count from model
        self.counts_model = np.zeros(self.shape)

        self.inter = None
        self.inter_ci = None
        self.supp = None
        self.supp_ci = None

    def update_model_counts(self, distr, N):
        """Update counts from model"""
        self.N_model += N
        self.counts_model += distr
    def update_data_counts(self, distr, N):
        """Update counts from data"""
        self.N_data += N
        self.counts_data += distr

    def tofile(self, file):
        file.write(str(self.N_model).encode().rjust(12))
        file.write(str(self.N_data).encode().rjust(12))
        self.counts_model.tofile(file)
        self.counts_data.tofile(file)
    def fromfile(self, f):
        self.__update_fields()
        #self.N_model = int(f.readline())
        #self.N_data = int(f.readline())
        self.N_model = int(f.read(12))
        self.N_data = int(f.read(12))
        cnt = math.prod(self.shape)
        self.counts_model = np.reshape(np.fromfile(f, float, cnt), self.shape)
        self.counts_data  = np.reshape(np.fromfile(f, float, cnt), self.shape)
    def clear_data(self):
        del self.shape
        del self.domsize
        del self.N_model
        del self.N_data
        del self.counts_model
        del self.counts_data
        

    def compute_interestingness(self, delta):
        self.inter = (np.abs(self.counts_model/self.N_model - self.counts_data/self.N_data)).max()
        if self.sample_from_data:
            z = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddevM = pM * (1.0 - pM) / self.N_model
            pD = self.counts_data / self.N_data
            stddevD = pD * (1.0 - pD) / self.N_data * (self.ND - self.N_data) / (self.ND - 1)
            stddev = stddevM + stddevD
            self.inter_ci = z * math.sqrt(stddev.max())
        else:
            #self.inter_ci = E_Chernoff(N_model, delta/self.domsize)
            #self.inter_ci = Utils.gaussinv.cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize) * 1.0 / (2*math.sqrt(N_model))
            # normal approx using different variances
            z = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            p = self.counts_model / self.N_model
            stddev = p * (1.0 - p)
            self.inter_ci = z * math.sqrt(stddev.max() / self.N_model)

    def compute_support(self, delta):
        self.supp = max(float(self.counts_data.max()) / self.N_data,
                        float(self.counts_model.max()) / self.N_model)
        if self.sample_from_data:
            z = cdf_ugaussian_Pinv(1.0 - 0.25*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddevM = pM * (1.0 - pM) / self.N_model
            pD = self.counts_data / self.N_data
            stddevD = pD * (1.0 - pD) / self.N_data * (self.ND - self.N_data) / (self.ND - 1)
            stddev = max(stddevM.max(), stddevD.max())
            self.supp_ci = z * math.sqrt(stddev)
        else:
            #self.supp_ci = E_Chernoff(self.N_model, delta/self.domsize)
            #self.supp_ci = Utils.gaussinv.cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize) * 1.0 / (2*math.sqrt(self.N_model))
            # normal approx using different variances
            z = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            p = self.counts_model / self.N_model
            stddev = p * (1.0 - p)
            self.supp_ci = z * math.sqrt(stddev.max() / self.N_model)

    def n_samples_required_inter(self, E, delta):
        """Returns the number of samples needed to get confidence
        interval for interestingness within E."""
        if self.sample_from_data:
            # normal approx using different variances
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddevM = pM * (1.0 - pM)
            pD = self.counts_data / self.N_data
            stddevD = pD * (1.0 - pD) * (self.ND - self.N_data) / (self.ND - 1)
            N = cdf_inv * cdf_inv * (stddevM + stddevD).max() / (E * E)
        else:
            N = N_Chernoff(E, delta/self.domsize)
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            N = 0.25 * cdf_inv * cdf_inv / (E * E)
            # normal approx using different variances
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddev = pM * (1.0 - pM)
            N = cdf_inv * cdf_inv * stddev.max() / (E * E)
        return N
    def n_samples_required_supp(self, E, delta):
        """Returns the number of samples needed to get confidence
        interval for support within E."""
        if self.sample_from_data:
            # normal approx using different variances
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.25*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddevM = pM * (1.0 - pM)
            pD = self.counts_data / self.N_data
            stddevD = pD * (1.0 - pD) * (self.ND - self.N_data) / (self.ND - 1)
            N = cdf_inv * cdf_inv * (stddevM.max(), stddevD.max()) / (E * E)
        else:
            N = N_Chernoff(E, delta/self.domsize)
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            N = 0.25 * cdf_inv * cdf_inv / (E * E)
            # normal approx using different variances
            cdf_inv = cdf_ugaussian_Pinv(1.0 - 0.5*delta/self.domsize)
            pM = self.counts_model / self.N_model
            stddev = pM * (1.0 - pM)
            N = cdf_inv * cdf_inv * stddev.max() / (E * E)
        return N

    @staticmethod
    def sort_key_inter(aset):
        """Compare attribute sets on interestingness.

        Used for sorting on interestingness in descending order."""
        return -aset.inter
    @staticmethod
    def sort_key_supp(aset):
        """Compare attribute sets on support.

        Used for sorting on support in descending order."""
        return -aset.supp



class BN_interestingness_sample(object):
    """Class for encapsulating the algorithm for fidning n most
    interesting attribute sets by prograssively sampling from both
    data and Bayesian network.

    if sample_from_data is False whole database will be read.""" 
    #def __init__(self, bn, ds, sample_from_data = False, ND = None):
    def __init__(self, bn, ds, sample_from_data = False, ND = 1000000):
        self.bn = bn
        self.ds = ds
        # chack attribute names match
        # TODO: reorder them
        if self.bn.get_attr_names() != self.ds.attrset.get_attr_names():
            raise RuntimeError("Attribute lists don't match. Dataset and BayesNet must have the same attribute names in the same order.")
        self.domainsizes = [len(a.domain) for a in self.ds.attrset]
        self.nattrs = len(self.ds.attrset)
        self.sample_from_data = sample_from_data
        if self.sample_from_data:
            if ND is None: # find database size
                ND = 0
                for x in self.ds:
                    ND += 1
                self.ds.rewind()
        self.ND = ND
        attr_set.sample_from_data = self.sample_from_data
        attr_set.ND = self.ND
        attr_set.domainsizes = self.domainsizes

        # map data attributes into network attributes
        self.data2net_attr_map = {}
        self.network_attrs_used = [] # numbers of network attributes used (also in data)
        for i, a in enumerate(self.ds.attrset):
            j = self.bn.names_to_numbers([a.name])[0]
            self.data2net_attr_map[i] = j
            self.network_attrs_used.append(j)

        self.temp_dir = tempfile.gettempdir()
        #self.temp_dir = "/tmp"
        #self.temp_dir = "/mnt/data"

    def run(self, n = 5, delta = 0.05, epsilon = 0.01, maxK = 4, disk_storage = True,
            excluded_attrs = [], selectionCond = None):
        ### parameters
        self.batch_size = 1000 # draw this number of samples at once
        self.n = n
        self.delta = delta
        self.epsilon = epsilon
        self.k = 1 # candidate attrset size.  used like apriori
        self.maxK = maxK # maximum attrset size
        self.disk_storage = disk_storage # should hypotheses be stored on disk (currently only distributions)
        self.debug = 1

        self.initialize(excluded_attrs, selectionCond)

        while len(self.C) > 0 or len(self.H) > n:
            ### if no more candidates and enough samples drawn, exit
            remaining_delta = sum([f(i) for i in range(1,self.niter)])
            if len(self.C) == 0 and self.minN_bn > 0 and self.inter_conf_interv_final(self.minN_bn, delta/len(self.H)*(1-2.0/3.0*remaining_delta), self.minN_data) < epsilon / 2: # don't need to take f(self.niter) into account here since half of delta has been allocated anyway
            #if len(self.C) == 0 and self.minN_bn > 0 and self.inter_conf_interv_final(self.minN_bn, delta/(3.0 * len(self.H))) < epsilon / 2: # don't need to take f(self.niter) into account here since half of delta has been allocated anyway
            #if len(self.C) == 0 and self.minN_bn > 0 and self.maxE < epsilon / 2: # don't need to take f(self.niter) into account here since 1/3 of delta has been allocated anyway
                break

            print("Iteration", self.niter, "-------------------------------------------------")
            #gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)

            hyp_batch = 30000 # how many hypotheses simultaneously?
            seg_begin = 0
            if self.sample_from_data:
                it = itertools.islice(self.ds, self.batch_size)
                sample_data = list(it)
            else:
                self.ds.rewind()
                sample_data = list(self.ds)
            # use iterative sampler
            #it = itertools.islice(self.sampler, self.batch_size)
            #sample_bn = list(it)
            # use array sampler (warning: record Selection is ignored)
            sample_bn = self.sampler1.draw_n_samples(self.batch_size).take(self.network_attrs_used, axis=1).tolist()
            
            if self.disk_storage:
                self.tmp_file.flush() ### needed on Windows!
                self.tmp_file.seek(0)
            while seg_begin < len(self.A):
                print("Counting hypotheses", seg_begin, "to", seg_begin + hyp_batch)
                if self.disk_storage:
                    offset = self.tmp_file.tell()
                    hyp = []
                    for h in self.A[seg_begin: seg_begin+hyp_batch]:
                        fullh = h.fromfile(self.tmp_file)
                        hyp.append(fullh)
                else:
                    hyp = self.A
                N = self.update_counts(hyp, sample_bn, sample_data)
                self.compute_interestingness(hyp, delta)
                self.compute_supports(hyp, delta)
                if self.disk_storage:
                    self.tmp_file.seek(offset) # assume hypotheses didn't change sizes
                    for i, fullh in enumerate(hyp):
                        h = self.A[seg_begin + i]
                        h.inter = fullh.inter
                        h.inter_ci = fullh.inter_ci
                        h.supp = fullh.supp
                        h.supp_ci = fullh.supp_ci
                        h.offset = self.tmp_file.tell()
                        fullh.tofile(self.tmp_file)
                        fullh.clear_data()
                seg_begin += hyp_batch
                self.tmp_file.flush() ### needed on Windows!

            if self.disk_storage:
                oldA = [h for h in self.A]
            #del sample_data
            #del sample_bn
            #del hyp
            self.minN_bn += N
            self.N_bn += N
            self.minN_data += N
            self.N_data += N


            LB, UB = self.compute_interestingness_bounds(n, delta, epsilon)
            minsT, npruned = self.prune(UB, delta, epsilon)

            prtLB = LB
            if LB is None:
                prtLB = 1e10
            print("N_bn=",self.N_bn,"self.minN_bn=",self.minN_bn, "|H|=%d" % len(self.H),\
                  "|C|=%d" % len(self.C), ("LB=%7.5f" % prtLB), ("UB=%7.5f" % UB))

            ### accept good candidates
            finish = False
            if len(self.C) == 0:
                finish = self.accept_good_candidates(LB, UB, epsilon)
            if finish:
                break
            ### reject bad candidates
            if len(self.C) == 0:
                self.reject_bad_candidates(UB, delta, epsilon)

            self.new_batch_size() # compute new batch_size based on a goal
            ### reject bad candidates
            # here we reject if we are still generating candidates.
            # If new candidates are generated we do pruning on previous hypotheses
            # this will never be executed together with pruning a few lines above
            if self.generate_cand and len(self.C) > 0:
                self.reject_bad_candidates(UB, delta, epsilon)

            ### remove pruned/rejected attr sets from disk
            if self.disk_storage:
                self.remove_pruned_rejected_from_disk(oldA)

            ### Generate new candidates
            self.candidates_generated = False
            if len(self.C) > 0 and self.k < self.maxK and self.k <= self.nattrs and self.generate_cand:
                self.generate_new_candidates(self.k)
                self.candidates_generated = True

            if self.k >= self.maxK:
                self.C = []

            #if self.candidates_generated == False: # don't do any testing if candidates were just generated, only do counting
            self.niter = self.niter + 1


        self.H.sort(key=attr_set.sort_key_inter)
        return [(h.key, h.inter) for h in self.H]
        #return self.H

    def remove_pruned_rejected_from_disk(self, oldA):
        Adict = {}
        for h in self.A:
            Adict[h.key] = None
        self.tmp_file.seek(0)
        tmp_file2 = open(self.temp_dir + "/bprune"+str(self.niter), "w+b")
        for h in oldA:
            fullh = h.fromfile(self.tmp_file)
            if h.key in Adict:
                h.offset = tmp_file2.tell()
                fullh.tofile(tmp_file2)
        self.tmp_file.close()
        os.remove(self.tmp_file.name)
        self.tmp_file = tmp_file2
        self.tmp_file.flush() ### needed on Windows!

    def new_batch_size(self):
        """Compute how many samples to draw now."""
        margin = 1.1
        new_bs = 5000
        #new_bs = self.batch_size

        if self.n > len(self.H):
            if len(self.C) > 0:
                self.generate_cand = True
            return

        def samples_required_inter(h1, h2, epsilon = 0.0, tmp_file = None):
            """How many samples are required (including extra margin)
            to make hypotheses i and j distinguishable by
            interestingness.

            a tolerance margin of epsilon is allowed."""
            # read distributions from disk if necessary
            if tmp_file is not None:
                tmp_file.seek(h1.offset)
                h1 = h1.fromfile(tmp_file)
                tmp_file.seek(h2.offset)
                h2 = h2.fromfile(tmp_file)
            E = (h1.inter - h2.inter + epsilon)/2
            bs1 = h1.n_samples_required_inter(E, self.delta*f(self.niter)/(3.0*len(self.H)))
            bs2 = h2.n_samples_required_inter(E, self.delta*f(self.niter)/(3.0*len(self.H)))
            bs = max(bs1, bs2)
            bs = int(bs*margin) - self.minN_bn
            return bs

        if self.disk_storage:
            tmp_file = self.tmp_file
        else:
            tmp_file = None
        if len(self.C) == 0:
            ind = self.n + int((len(self.H) - self.n) * .75)
            bs_reject = samples_required_inter(self.H[self.n-1], self.H[ind], tmp_file = tmp_file)
            print("new batch size needed to reject", bs_reject)
            bs_accept = samples_required_inter(self.H[self.n-1], self.H[self.n], self.epsilon, tmp_file = tmp_file)
            print("new batch size needed to accept", bs_accept)
            new_bs = max(10000, min(bs_accept, bs_reject))
        elif len(self.C) > 0:
            ind = int(len(self.H) * .9)
            Hs = list(self.H)
            Hs.sort(key=attr_set.sort_key_supp)
            print("--->%f,%f"%(Hs[ind].supp , self.H[self.n-1].inter))
            print(" ".join([str(h.supp) for h in Hs[-50:]]))
            if Hs[ind].supp < self.H[self.n-1].inter:
                h1 = self.H[self.n-1]
                h2 = Hs[ind]
                if tmp_file is not None:
                    tmp_file.seek(h1.offset)
                    h1 = h1.fromfile(tmp_file)
                    tmp_file.seek(h2.offset)
                    h2 = h2.fromfile(tmp_file)
                E = (h1.inter - h2.supp)/2
                bs1 = h1.n_samples_required_inter(E, self.delta*f(self.niter)/(3.0*len(self.H)))#, tmp_file = tmp_file)
                bs2 = h2.n_samples_required_supp(E, self.delta*f(self.niter)/(3.0*len(self.H)))#, tmp_file = tmp_file)
                del Hs
                bs = max(bs1, bs2)
                bs = int(bs*margin) - self.minN_bn
                bs = max(1000, bs)
                if bs < len(self.C) * self.nattrs:
                    new_bs = bs
                    self.generate_cand = False
                    print("new batch size needed to prune", new_bs)
                else:
                    self.generate_cand = True
            else:
                self.generate_cand = True

        self.batch_size = new_bs

    def initialize(self, excluded_attrs, selectionCond):
        """Initialize algorithm parameters and data structures"""
        if self.debug >= 1:
            print("Initializing algorithm")

        ### initialize candidate frequent attribute sets
        self.A = [] # attribute sets which are still counted
        self.H = [] # attribute sets which could still be interesting
        self.C = [] # attribute sets which are frequent and could generate more candidates
                    # but could already be rejected as uninteresting

        ### initialize counts
        self.N_bn = 0  # total number of samples drawn from BN
        self.minN_bn = 0 # smallest # of samples for any attrset.  Needed to know when to stop iteration
        self.maxE = 10e10 # maximum interestingness estimation error (over all attribute sets in H)
        self.N_data = 0 # total number of samples drawn from data
        self.minN_data = 0 # smallest # of samples for any attrset.  Needed to know when to stop iteration

        self.niter = 1 # number or iteration

        # initial candidates
        if self.disk_storage:
            self.tmp_file = open(self.temp_dir + "/bprunedata", "w+b")
        for i, a in enumerate(self.ds.attrset):
            if a.name not in excluded_attrs:
                h = attr_set((i,))
                if self.disk_storage:
                    h.tofile(self.tmp_file)
                    h = small_attr_set(h)
                self.A.append(h)
                self.C.append(h.key)
                self.H.append(h)
        self.candidates_generated = True # flag stating that candidates were just generated
        self.generate_cand = False # whether new candidates should be generated on this iteration
        #self.counted_on_all_data = False # whether we completed counting from data

        #self.sampler = BayesSampler(self.bn)  # class for sampling from bayesian network
        self.sampler1 = BayesSampler(self.bn)
        self.sampler2 = SelectionReader(self.sampler1, selectionCond)
        self.sampler3 = ProjectionReader(self.sampler2, self.network_attrs_used)
        self.sampler = self.sampler3 # class for sampling from bayesian network


    def update_counts(self, hyp, sample_bn, sample_data):
        """Draw more samples and update counts for attribute sets in hyp"""
        if self.debug >= 1:
            print("Updating counts")

        domsizes = self.domainsizes#[len(d) for d in self.bn.attrdomains]
        asets = [h.key for h in hyp]

        ### count from data
        if self.sample_from_data:
            print("sampling from data")
            #counts, N, missing_counts = Utils.Counts.compute_counts_array_cover(asets, self.ds, self.nattrs, domsizes, maxN = self.batch_size)
            counts, N, missing_counts = compute_counts_array_cover(asets, sample_data, self.nattrs, domsizes, maxN = self.batch_size)
            for h in hyp:
                h.update_data_counts(counts[h.key], N - missing_counts[h.key])
            #self.minN_data += N
            #self.N_data += N
        else:
            if self.candidates_generated:
                self.ds.rewind()
                sample_data = list(self.ds)
                counts_data, N_data, missing_counts = compute_counts_array_cover(asets, sample_data, self.nattrs, domsizes, maxN = -1)
                for h in hyp:
                    h.N_data = N_data - missing_counts[h.key]
                    h.counts_data = counts_data[h.key]

        ### count from BN
        #counts, N = Utils.Counts.compute_counts_array_cover(asets, self.sampler, self.nattrs, domsizes, maxN = self.batch_size)
        counts, N, missing_counts = compute_counts_array_cover(asets, sample_bn, self.nattrs, domsizes, maxN = self.batch_size)
        for h in hyp:
            h.update_model_counts(counts[h.key], N - missing_counts[h.key])
        #self.minN_bn += N
        #self.N_bn += N
        return N

    def compute_interestingness(self, hyp, delta):
        """Compute interestingness of attribute sets in hyp"""
        if self.debug >= 1:
            print("Computing interestingness")
        for h in hyp:
            h.compute_interestingness(delta*f(self.niter)/(3.0*len(self.H)))

    def compute_supports(self, hyp, delta):
        """compute supports of attribute sets in hyp."""
        if self.debug >= 1:
            print("Computing supports")
        for h in hyp:
            h.compute_support(delta*f(self.niter)/(3.0*len(self.H)))


    def compute_interestingness_bounds(self, n, delta, epsilon):
        """Compute interestingness related bounds"""
        self.H.sort(key=attr_set.sort_key_supp)
        self.maxE = max([h.inter_ci for h in self.H])
        if len(self.H) > n:
            LB = max([h.inter + h.inter_ci for h in self.H[n:]])
        else:
            LB = None
        UB = min([h.inter - h.inter_ci for h in self.H[:n]])
        return LB, UB

    def prune(self, UB, delta, epsilon):
        """Prune attribute sets with support significantly smaller
        then interestingness of n-th most interesting attribute set.
        Returns minimum support of all attrsets corrected for
        statistical errors."""
        if self.debug >= 1:
            print("Pruning")
        deleted = {}
        corrected_minsup = 10e10
        for h in self.H:
            if h.supp < UB - h.supp_ci:
                deleted[h.key] = h.supp
            if h.supp + h.supp_ci < corrected_minsup:
                corrected_minsup = h.supp + h.supp_ci
        print("min support", corrected_minsup)
        # delete supersets - not necessary, they will be deleted anyway!!!
        #delsetree = Utils.SEtree.SEtree()
        #for aset in deleted.iterkeys():
        #    delsetree[aset] = None
        #deleted = {}
        #for h in self.H:
        #    if len(list(delsetree.iter_included(h.key))) > 0:
        #        deleted[h.key] = None
        #for aset in self.C:
        #    if len(list(delsetree.iter_included(aset))) > 0:
        #        deleted[aset] = None
        self.A = [x for x in self.A if x.key not in deleted]
        self.H = [x for x in self.H if x.key not in deleted]
        self.C = [x for x in self.C if x not in deleted]
        print(len(deleted), "attrsets pruned")
        #for aset in deleted.iterkeys():
        #    print("infrequent", aset)
        return(corrected_minsup, len(deleted))


    def accept_good_candidates(self, LB, UB, epsilon):
        if LB is not None and UB >= LB - epsilon:
            print("!!!!!Good ones accepted!!!!!")
            return True
        return False

    def reject_bad_candidates(self, UB, delta, epsilon):
        if self.debug >= 1:
            print("Rejecting bad candidates")
        deleted = {}
        for h in self.H:
            if len(self.H) - len(deleted) > self.n:
                if h.inter <= UB - h.inter_ci:
                    #print("bad", h.key, h.inter)
                    deleted[h.key] = None
        print(len(deleted), "attrsets rejected")
        self.A = [x for x in self.A if x.key not in deleted]
        self.H = [x for x in self.H if x.key not in deleted]

    def generate_new_candidates(self, k):
        if self.debug >= 1:
            print("Generating new candidates")
        def join_itemsets(joinable, newC):
            for x in range(len(joinable)):
                left = joinable[x]
                for right in joinable[x+1:]:
                    newC.append(left + (right[-1],))

        self.minN_bn = 0 # new candidates are not counted at all
        newC = []
        it = iter(self.C)
        first = next(it)
        joinable = [first]
        for iset in it:
            if first[0:self.k-1] == iset[0:self.k-1]:
                joinable.append(iset)
            else:
                join_itemsets(joinable, newC)
                first = iset
                joinable = [iset]
        join_itemsets(joinable, newC)
        ### Prune new candidates with infrequent subests
        deleted = {}
        Cdict = {}
        for c in self.C:
            Cdict[c] = None
        for c in newC:
            for i in range(len(c) - 2):
                cc = c[0:i]+c[i+1:]
                if cc not in Cdict:
                    deleted[c] = None
                    break
        del Cdict
        newC = [x for x in newC if x not in deleted]

        #print newC
        if self.disk_storage:
            self.tmp_file.seek(0, 2)
        self.C = newC
        i = 0
        for aset in newC:
            h = attr_set(aset)
            if self.disk_storage:
                h.tofile(self.tmp_file)
                h.clear_data()
                h = small_attr_set(h)
            self.H.append(h)
            self.A.append(h)
            i = i + 1
            if i % 10000 == 0:
                print(i)

        self.tmp_file.flush() ### needed on Windows!
        self.k = self.k + 1
        
    ### supplementary functions
    def __domsize(self, aset):
        domsize = 1
        for a in aset:
            domsize *= self.domainsizes[a]
        return domsize

    def inter_conf_interv_final(self, Nbn, delta, Ndata = None):
        totsize = 0
        for h in self.H:
            totsize += self.__domsize(h.key)
        #ci = E_Chernoff(N, delta/totsize)
        if self.sample_from_data:
            ci = cdf_ugaussian_Pinv(1.0 - 0.5*delta/totsize) \
                 * 0.5 * math.sqrt(1.0 / Nbn + 1.0 / Ndata * (self.ND - Ndata) / (self.ND - 1))
        else:
            ci = cdf_ugaussian_Pinv(1.0 - 0.5*delta/totsize) * 1.0 / (2*math.sqrt(Nbn))
        return ci


    def compute_attrset_interestingness(self, aset):
        h = None
        for h2 in self.H:
            if h2.key == aset:
                h = h2
                break
        if h is None:
            raise RuntimeError("attrset not found")

        if self.disk_storage:
            tmp_file = self.tmp_file
        else:
            tmp_file = None
        if tmp_file is not None:
            tmp_file.seek(h.offset)
            h = h.fromfile(tmp_file)

        h.compute_interestingness(0.05)
        return h.inter, h.counts_data / h.N_data, h.counts_model / h.N_model
