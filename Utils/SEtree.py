from __future__ import generators
from copy import copy

class SEtree(object):
    """Set Enumeration tree class"""
    def __init__(self):
        self.len = 0
        self.root = SEtree_node()
    def __len__(self):
        return self.len
    def __setitem__(self, set, item):
        self.__ensure_sorted(set)
        if self.root.add(set, item):
            self.len += 1
    def __getitem__(self, set):
        cur = self.__get_node(set)
        if not cur.has_item:
            raise KeyError("SEtree: " + repr(set))
        return cur.item
    def __contains__(self, set):
        self.__ensure_sorted(set)
        try:
            self.__getitem__(set)
        except KeyError:
            return False
        return True
    def __delitem__(self, set):
        cur = self.__get_node(set)
        if cur.has_item is False:
            raise KeyError("SEtree: " + repr(set))
        cur.item = None
        cur.has_item = False

        # remove unused nodes
        while cur is not None and not cur.has_item and cur.children is None and cur.parent is not None:
            del cur.parent.children[set[-1]]
            if len(cur.parent.children) == 0:
                cur.parent.children = None
            set = set[:-1]
            cur = cur.parent
            
        self.len -= 1
        

    def __iter__(self):
        return self.__iter_generator(self.root, [])
    def __iter_generator(self, node, prefix):
        if node.has_item:
            yield tuple(prefix)
        if node.children is not None:
            keys = list(node.children.keys())
            keys.sort()
            for e in keys:
                prefix.append(e)
                for k in self.__iter_generator(node.children[e], prefix):
                    yield k
                prefix.pop()

    def items(self):
        return self.__iteritems_generator(self.root, [])
    def __iteritems_generator(self, node, prefix):
        if node.has_item:
            yield (tuple(prefix), node.item)
        if node.children is not None:
            keys = list(node.children.keys())
            keys.sort()
            for e in keys:
                prefix.append(e)
                for k in self.__iteritems_generator(node.children[e], prefix):
                    yield k
                prefix.pop()

    #def iter_included(self, superset):
    #    return self.__iter_included_generator(self.root, superset, [])
    def __iter_included_generator(self, node, superset, prefix):
        if node.has_item:
            yield tuple(prefix)
        if node.children is not None:
            for e in superset:
                if e in node.children:
                    prefix.append(e)
                    for k in self.__iter_included_generator(node.children[e], superset, prefix):
                        yield k
                    prefix.pop()

    def iter_included(self, superset):
        return self.included_iterator(self.root, superset)
    class included_iterator(object):
        def __init__(self, node, superset):
            self.stack = [(node, [])]
            self.superset = superset
        def __iter__(self):
            return self
        def __next__(self):
            while True:
                if len(self.stack) == 0:
                    raise StopIteration
                node, prefix = self.stack.pop()
                if node.children is not None:
                    for e in self.superset:
                        if e in node.children:
                            self.stack.append((node.children[e], prefix + [e]))
                if node.has_item:
                    return tuple(prefix)


    def iter_one_not_in(self, superset):
        """Iterate all subsets with exactly one element not in
        superset.  Returns the set and the element not in superset.

        Used for finding covers of attribute sets."""
        return self.one_not_in_iterator(self.root, superset)
    class one_not_in_iterator(object):
        def __init__(self, node, superset):
            self.stack = [(node, [], 0, None)]
            self.superset = superset
            self.superset_dict = {}
            for x in superset:
                self.superset_dict[x] = None
        def __iter__(self):
            return self
        def __next__(self):
            while True:
                if len(self.stack) == 0:
                    raise StopIteration
                node, prefix, not_in, elem_not_in = self.stack.pop()
                if node.children is not None:
                    if not_in == 1:
                        for e in self.superset:
                            if e in node.children:
                                self.stack.append((node.children[e], prefix + [e], not_in, elem_not_in))
                    else:
                        for e in node.children.keys():
                            if e in self.superset_dict:
                                self.stack.append((node.children[e], prefix + [e], not_in, None))
                            else:
                                self.stack.append((node.children[e], prefix + [e], not_in + 1, e))
                if not_in == 1 and node.has_item:
                    return tuple(prefix), elem_not_in
        




    def get(self, k, x = None):
        if k in self:
            return self[k]
        return x
    def setdefault(self, k, x=None):
        if k not in self:
            self[k] = x
        return self[k]
    def popitem(self):
        if len(self) == 0:
            raise KeyError("SEtree is empty")
        item = next(self.items())
        del self[item[0]]
        return item
    def keys(self):
        return [k for k in self]
    #def values(self):
    #    return [i[1] for i in self.iteritems()]
    def values(self):
        it = self.iteritems()
        for item in it:
            yield item[1]
    def clear(self):
        self.len = 0
        self.root = SEtree_node()
    def copy(self):
        return copy(self)
    def update(self, b):
        for k in b.keys():
            self[k] = b[k]
    def __str__(self):
        ret = "SE Tree:\n"
        ret += "\n".join([str(x)+" --> "+str(self[x]) for x in self])
        return ret

    def __get_node(self, set):
        """Return the node of the tree corresponding to set"""
        self.__ensure_sorted(set)
        cur = self.root
        try:
            for e in set:
                cur = cur.children[e]
        except (KeyError, AttributeError):
            raise KeyError("SEtree: " + repr(set))
        return cur
    def __ensure_sorted(self, seq):
        """Ensure that seq is sorted, raise KeyError otherwise"""
        for i in range(len(seq) - 1):
            if(seq[i] > seq[i+1]):
                raise KeyError("key " + repr(seq) + " is not sorted")

        

class SEtree_node(object):
    """Set Enumeration tree node"""
    def __init__(self):
        self.item = None                # value in this node
        self.children = None            # initially no children
        self.has_item = False           # no item here yet
        self.parent = None

    def add(self, set, item):           # internal method to add a set to a tree
        if len(set) == 0:
            self.item = item
            ret = not self.has_item
            self.has_item = True
            return ret
        if self.children is None:
            self.children = {}
        if set[0] in self.children:
            child = self.children[set[0]]
        else:
            child = SEtree_node()
            child.parent = self
            self.children[set[0]] = child
        return child.add(set[1:], item)


if __name__ == "__main__":
    s = SEtree()
    print(s)
    s[[0,1]] = "x"
    print(s[0,1])
    try:
        print(s[5,])
    except KeyError:
        print("incorrect key")
    else:
        print("Error: incorrect key did not cause exception")
    print("contains(5) = " + str((5,) in s))
    print("contains(0,1) = " + str((0,1) in s))

    s[0,] = "a"
    s[()] = "empty"
    s[2,] = "b"
    print()
    print(s)
    print()
    superset = (2,)
    print("Iteration over sets included in",superset,":")
    for x in s.iter_included(superset):
        print(x)
    iter = SEtree.included_iterator(s.root, superset)
    for x in iter:
        print(x)
    print("iter one_not_in {2}")
    iter = SEtree.one_not_in_iterator(s.root, superset)
    for x in iter:
        print(x)
    print("iter one_not_in {0}")
    iter = SEtree.one_not_in_iterator(s.root, (0,))
    for x in iter:
        print(x)


    del s[2,]
    print()
    print(s)
    print("keys:", s.keys())
    print("items:", s.items())
    print("values:", s.values())
    print("(2,) in s:", (2,) in s)

    print("popitem:",s.popitem())
    print(s)
    print("itervalues:", [v for v in s.values()])
