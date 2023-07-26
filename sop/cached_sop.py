from .sop import array_sop


max_cached =     20000000
max_cache_size = 100000000

class cached_array_sop(array_sop):
    """An array sum of products using dynamic programming.

    Works like an array_sop but uses memoization to reuse partial sums
    over several summations."""

    class factor(array_sop.factor):
        def compute_key(self):
            return set()
    class bucket(array_sop.bucket):
        def cost(self):
            key = self.compute_key()
            if key in self.sop.sum_cache:
                return 0, 0
            return array_sop.bucket.cost(self)
            
        def compute(self, values):
            key = self.compute_key()
            if key in self.sop.sum_cache:
                #print "hit:", key
                return self.sop.sum_cache[key]
            array = array_sop.bucket.compute(self, values)
            # avoid caching huge arrays
            arrsize = array.nbytes
            if arrsize < max_cached:
                if self.sop.cached_size + arrsize > max_cache_size:
                    self.sop.clear_cache()
                self.sop.sum_cache[key] = array
                self.sop.cached_size += arrsize
            return array
        def compute_key(self):
            """Computes the key for memoization.

            The key is the variable in the bucket union the variables
            in its children."""
            key = set([self.var])
            for p in self.product:
                key = key | p.compute_key()
            key = frozenset(key)
            return key
    def __init__(self, n):
        """Create an array sop with n variables"""
        super().__init__(n)
        self.sum_cache = {}
        self.cached_size = 0
    def clear_cache(self):
        self.sum_cache.clear()
        self.cached_size = 0
