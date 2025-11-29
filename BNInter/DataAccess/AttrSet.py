"""Functions and classes common to accessing data files."""



# missing values

class missing(object):
    """A class representing a missing value"""
    def __str__(self):
        return "?"
    def __repr__(self):
        return "?"

def ismissing(v):
    """Checks if v is a missing value."""
    return isinstance(v, missing)


# attributes

attrtypes = ["CATEG","CONTINUOUS","STRING"]




class Attr(object):
    """An attribute."""
    def __init__(self, name, attr_type, domain = None):
        self.name = name
        self.type = attr_type
        self.domain = domain
        if self.type not in attrtypes:
            raise RuntimeError("Wrong attribute type")
        if self.type == "CATEG" and self.domain is None:
            self.domain = []

    def __str__(self):
        ret = self.name + ":"
        if self.type == "CATEG":
            ret += str(self.domain)
        else:
            ret += self.type
        return ret


# sets of attributes

class AttrSet(object):
    def __init__(self, name = "", attrs = None):
        self.name = name
        if attrs is None:
            attrs = []
        self.attrs = attrs

        self.names_2_numbers_map = {}
        for i, a in enumerate(self.attrs):
            self.names_2_numbers_map[a.name] = i


    # accessors
    def __len__(self):
        return len(self.attrs)
    def __iter__(self):
        return iter(self.attrs)
    def __getitem__(self, attr_index):
        if isinstance(attr_index, str):
            return self.attrs[self.names_2_numbers_map[attr_index]]
        else:
            return self.attrs[attr_index]

    def names_to_numbers(self, attrnames):
        """Converts a list of attribute names to a list of attribute
        numbers."""
        if isinstance(attrnames, str):
            raise RuntimeError("names_to_numbers: need a list of attribute names")
        return [self.names_2_numbers_map[aname] for aname in attrnames]
    def numbers_to_names(self, attrnumbers):
        """Converts a list of attribute numbers to a list of attribute
        names."""
        return [self.attrs[anumber].name for anumber in attrnumbers]

    def get_attr_names(self):
        """Return names of all nodes in order."""
        return [attr.name for attr in self.attrs]

    def append_attr(self, attr):
        self.attrs.append(attr)
        self.names_2_numbers_map[attr.name] = len(self) - 1

    def __str__(self):
        ret = ""
        ret += "AttrSet: " + self.name + "\n"
        ret += "\n".join([str(a) for a in self])
        return ret



if __name__ == "__main__":
    aset = AttrSet("test", [Attr("a1", "STRING")])
    print(aset)
    print()
    aset.append_attr(Attr("a2", "CATEG", ["v1", "v2", "v3"]))
    aset.append_attr(Attr("a3", "CONTINUOUS"))
    aset.append_attr(Attr("a4", "CATEG", ["a", "b", "c"]))
    print(aset)

    print()
    print(aset.names_to_numbers([a.name for a in aset]))
    print(aset.numbers_to_names(range(len(aset))))
    
