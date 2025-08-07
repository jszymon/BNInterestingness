import copy


class sop_base(object):
    """Sum of products base class."""
    class factor(object):
        """A factor in the sum of products"""
        def __init__(self, variables):
            self.vars = variables
        def compute(self, values):
            return None
        def cost(self):
            return (0,0)
        def complexity(self):
            return 0
        def result_size(self):
            return 0
        def mem_cost(self):
            return 0
        def bucket_elim(self, parent):
            pass
        def str_rec(self, depth):
            ret = " " * (depth * 4)
            ret += "f" + str(tuple(self.vars))
            if self.mem_cost() > 0:
                ret += " mem=" + str(self.mem_cost())
            return ret

    class bucket(object):
        """Structure representing a single summation variable."""
        def __init__(self, var, sop):
            """Create a new sop bucket, which sums over variable var."""
            self.var = var
            self.sop = sop # pointer to the parent sop
            self.product = []
            self.vars = []#which vars the bucket+children depend on
        def compute(self, values):
            return None
        def cost(self):
            """How many (adds, muls) are needed."""
            return (None, None)
        def complexity(self):
            return None
        def result_size(self):
            return 0
        def mem_cost(self):
            return 0
        def bucket_elim(self, parent):
            # do elimination in child buckets first
            for f in self.product:
                f.bucket_elim(self)
            # move children from b to parent if possible
            if parent is not None:
                for f in copy.copy(self.product):
                    if self.var not in f.vars:
                        parent.product.append(f)
                        self.product.remove(f)
            # compute on which vars this bucket depends
            self.vars = []
            for f in self.product:
                for v in f.vars:
                    if v not in self.vars:
                        self.vars.append(v)
            self.vars = [v for v in self.vars if v != self.var]
            self.vars.sort()
        def str_rec(self, depth):
            ret = " " * (depth * 4)
            if self.var != -1:
                ret += "Sum over variable " + str(self.var)
            else:
                ret += "Dummy bucket"
            ret += " function of " + str(self.vars)
            ret += " cost=" + str(self.cost())
            if self.mem_cost() > 0:
                ret += " mem=" + str(self.mem_cost())
            #ret += " reslt.size=" + str(self.result_size())
            #ret += "Value: " + str(self.compute(None)) + "\n"
            ret += "\n"
            ret += "\n".join([p.str_rec(depth + 1) for p in self.product])
            return ret


    def __init__(self, n):
        """Create a sop with n variables"""
        self.n = n
        self.product = []
    def add_factor(self, variables, *args):
        """Add a factor to the sop."""
        self.product.append(self.factor(variables, *args))
    def compute(self):
        """Compute the value."""
        values = [None] * self.n
        return self.prepared.compute(values)
    def cost(self):
        """Return a tuple containing number of additions and
        multiplications needed to compute the sop."""
        return self.prepared.cost()
    def complexity(self):
        """Return the exponent d of the O(k^d) time complexity
        of the computation of the sop."""
        return self.prepared.complexity()
    def mem_cost(self):
        """Return the maximum amount of memory used while computing the sop."""
        return self.prepared.mem_cost()
    def result_size(self):
        """Compute the size of the result returned."""
        return self.prepared.result_size()
    def prepare(self, *args):
        self.permutation = range(self.n) # permutation of variables
        self.order_variables(*args)
        #print "Ordering: " + str(self.permutation)
        self.__create_buckets(*args)
        self.__bucket_elimination()

    def __create_buckets(self, *args):
        """Create summation buckets for a configuration given by *args.

        By default create a dummy bucket (with variable number -1)
        and a bucket for every variable in the order of self.permutation.
        The last inserted buckets will contain all the factors."""
        # Create a 'dummy header' summation
        self.prepared = self.bucket(-1, self, *args)
        # insert a bucket for every variable
        last_bucket = self.prepared
        for v in self.permutation:
            new_bucket = self.bucket(v, self, *args)
            last_bucket.product.append(new_bucket)
            last_bucket = last_bucket.product[0]
        # insert factor buckets
        last_bucket.product = copy.copy(self.product)
    def order_variables(self, *args):
        """Heuristically order vairables to reduce computation cost"""
        # sort variables by decreasing number of factors depending on them
        deps = [[0, i] for i in range(self.n)]
        for f in self.product:
            for v in f.vars:
                deps[v][0] += 1
        deps.sort()
        deps.reverse()
        self.permutation = [x[1] for x in deps]
    def __bucket_elimination(self):
        """Perfrom elimination: move factors before summations"""
        self.prepared.bucket_elim(None)
    def __str__(self):
        return self.prepared.str_rec(0)
