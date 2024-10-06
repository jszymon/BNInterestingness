import copy
import numpy


from .sop_base import sop_base


def sort_axes(variables, array):
    """Sorts the variables vector and transposes array's axes
    accordingly.  Returns the tuple:
    (sorted vector, transposed array)."""
    tmp = list(zip(variables, range(len(variables))))
    tmp.sort()
    new_variables = [x[0] for x in tmp]
    new_axes = [x[1] for x in tmp]
    return (new_variables, numpy.transpose(array, new_axes))

class simple_sop(sop_base):
    """Simple sum of products implementation.

    Useful for obtaining a single probability from a marginal
    distribution.

    Arithmetic addition and multiplication is used, all variables are
    discrete and are assumed to take integer values starting at 0.
    Summation returns a single value, the sum is performed with
    specified bounds for each variable.  Memory consumption is always
    small, but certain sub-sums may be recomputed."""

    class factor(sop_base.factor):
        def __init__(self, variables, array):
            new_variables, new_array = sort_axes(variables, array)
            sop_base.factor.__init__(self, new_variables)
            self.array = new_array
        def compute(self, values):
            projected_values = tuple([values[i] for i in self.vars])
            return self.array[projected_values]
    class bucket(sop_base.bucket):
        def __init__(self, var, sop, min, max):
            """Create a new sop bucket, which sums over variable var ranging from min to max.
            
            var=-1 means a dummy bucket without any variable.
            Used to represent the whole expression."""
            sop_base.bucket.__init__(self, var, sop)
            if var == -1:
                # dummy bucket
                self.min = 0
                self.max = 0
            else:
                self.min = min[var]
                self.max = max[var]
        def compute(self, values):
            """Compute the value."""
            sum = 0.0
            for v in range(self.min, self.max + 1):
                values[self.var] = v
                prod = 1.0
                for p in self.product:
                    prod *= p.compute(values)
                sum += prod
            return sum
        def cost(self):
            """How many adds/muls are needed."""
            n = self.max - self.min + 1
            add_cost = n
            mul_cost = n * len(self.product)
            for p in self.product:
                tmp_cost = p.cost()
                add_cost += n * tmp_cost[0]
                mul_cost += n * tmp_cost[1]
            return (add_cost, mul_cost)
        def complexity(self):
            complexity = 0
            for p in self.product:
                complexity = max(complexity, p.complexity())
            if self.var != -1 and self.max > self.min:
                complexity += 1
            return complexity
        

class array_sop(sop_base):
    """A sum of products implementation returning and array.

    Useful for obtaining full marginal distributions.

    Arithmetic addition and multiplication is used, all variables are
    discrete and are assumed to take integer values starting at 0.
    The sum is performed over all variables except specified free
    variables.  Returned is the array being a function of the free
    variables.  The axes in the returned array are in the order
    specified in the argument to prepare method.

    Implements the bucket elimination procedure described by Dechter.
    Memory consumption may be exponential, but certain sub-sums will
    not be recomputed."""

    class factor(sop_base.factor):
        def __init__(self, variables, array):
            new_variables, new_array = sort_axes(variables, array)
            sop_base.factor.__init__(self, new_variables)
            self.array = new_array
        def compute(self, values):
            return self.array 
        def result_size(self):
            """See below."""
            size = 1
            for l in self.array.shape:
                size *= l
            return size
        def mem_cost(self):
            return self.array.nbytes
    class bucket(sop_base.bucket):
        def __init__(self, var, sop, free_vars):
            """Create a new sop bucket, which sums over variable var.
            
            var=-1 means a dummy bucket without any variable.
            Used to represent the whole expression."""
            sop_base.bucket.__init__(self, var, sop)
        def compute(self, values):
            """Compute the value."""
            prod = numpy.array(1)
            all_vars = []
            # sort products on size for faster mults
            l = [(x.result_size(), x) for x in self.product]
            l.sort(key = lambda x: x[0])
            sp = [x[1] for x in l]
            for p in sp:
                tmp_array = p.compute(values)
                old_all_vars = copy.copy(all_vars)
                for v in p.vars:
                    if v not in all_vars:
                        all_vars.append(v)
                all_vars.sort()
                # find new shape of tmp_array to reorder variables
                shape = [1] * len(all_vars)
                for v in p.vars:
                    shape[all_vars.index(v)] = self.sop.dim[v]
                tmp_array = numpy.reshape(tmp_array, shape)
                # find new shape of prod to reorder variables
                shape_prod = [1] * len(all_vars)
                for v in old_all_vars:
                    shape_prod[all_vars.index(v)] = self.sop.dim[v]
                prod = prod.reshape(shape_prod)
                prod = prod * tmp_array
            #sum over the bucket variable
            if self.var != -1:
                prod = numpy.sum(prod, all_vars.index(self.var))
            prod = numpy.array(prod) # force scalars to be returned as arrays
            return prod
        def cost(self):
            """How many adds/muls are needed."""
            add_cost = 0
            mul_cost = 0
            dim = self.sop.dim
            for p in self.product:
                tmp_cost = p.cost()
                add_cost += tmp_cost[0]
                mul_cost += tmp_cost[1]
            # cost within this bucket
            size = 1
            for v in self.vars:
                size *= dim[v]
            if self.var != -1:
                add_cost += (dim[self.var] - 1) * size
                size *= dim[self.var]
            mul_cost = mul_cost + size * len(self.product)
            return (add_cost, mul_cost)
        def result_size(self):
            """Size of the result this bucket returns."""
            dim = self.sop.dim
            size = 1
            for v in self.vars:
                size *= dim[v]
            return size
            
        def mem_cost(self):
            mem_cost = 0
            max_child = 0
            dim = self.sop.dim
            for p in self.product:
                mem_cost += p.result_size() * 8 #TODO: don't use hard coded const
                max_child = max(p.mem_cost(), max_child)
            # cost within this bucket
            size = self.result_size()
            if self.var != -1:
                size *= dim[self.var]
            mem_cost = mem_cost + size * 8 # TODO: get element size from array somehow
            return max(max_child, mem_cost)
            
        def complexity(self):
            """The exponent is the largest number of
            variables on which a bucket depends."""
            complexity = len(self.vars)
            if self.var != -1:
                complexity += 1
            for p in self.product:
                complexity = max(complexity, p.complexity())
            return complexity

    def __init__(self, n):
        """Create an array sop with n variables"""
        sop_base.__init__(self, n)
        self.dim = [1] * n
    def add_factor(self, variables, array):
        """Add a factor to the sop."""
        sop_base.add_factor(self, variables, array)
        for v, d in zip(variables, array.shape):
            self.dim[v] = d
    def prepare(self, free_vars):
        self.free_vars = free_vars
        sop_base.prepare(self, free_vars)
    def compute(self):
        val = sop_base.compute(self)
        # reorder axes
        new_axes = [-1] * len(self.free_vars)
        tmp = copy.copy(self.free_vars)
        tmp.sort()
        for i in range(len(new_axes)):
            v = tmp[i]
            new_axes[i] = self.free_vars.index(v)
        return numpy.transpose(val, new_axes)
        
    def order_variables(self, free_vars):
        """Heuristically order vairables to reduce computation cost"""
        # sort variables by decreasing number of factors depending on them
        sop_base.order_variables(self)
        self.permutation = [x for x in self.permutation if x not in free_vars]
