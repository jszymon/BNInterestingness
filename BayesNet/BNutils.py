import numpy
import itertools

#utilities
def blockiter(seq, n):
    it = iter(seq)
    while True:
        ret = list(itertools.islice(it, n))
        if len(ret) == 0:
            break
        yield ret

def distr_2_str(distr, cond = False):
    # numarray.__str__ seems slow and doesn't work with psyco
    ret = ""
    if not cond:
        ind = numpy.ndindex(distr.shape)
    else:
        ind = numpy.ndindex(distr.shape[:-1])
    for i in ind:
        d = distr[i]
        if not cond:
            dstr = "%g" % d
        else:
            dstr = "[" + " ".join(["%g" % x for x in d]) + "]"
        ret += str(i) + "->" + dstr  + "\n"
    #ret = "\n".join([str(i) + "->" + str(distr[i]) for i in ind])
    #ret += "\n"
    return ret

    
