from .AttrSet import Attr, AttrSet


class RecordReader(object):
    """An abstract record reader class."""
    def __init__(self, attrset):
        """Initialize the record reader.

        attrset - set of attributes (an instance of AttrSet class)"""
        self.attrset = attrset
        self.dummy_rec = [None] * len(self.attrset) # the dummy record to return

    def get_attr_set(self):
        return self.attrset

    def __iter__(self):
        return self
    def next(self):
        """Return the next record."""
        return self.dummy_rec
    def rewind(self):
        """Go back to the beginning."""
        pass

    def str_record(self, record, quote_char = ""):
        """Pretty print a record."""
        r_str = []
        for a, v in zip(self.attrset, record):
            if v is None:
                vstr = "?"
            elif a.type == "CATEG":
                vstr = quote_char + a.domain[v] + quote_char
            else:
                vstr = quote_char + str(v) + quote_char
            r_str.append(vstr)
        return ",".join(r_str)


if __name__ == "__main__":
    rr = RecordReader(AttrSet("dummy", [Attr("a1", "STRING"), Attr("a2", "CATEG", ["a", "b", "c"])]))
    for i in range(5):
        print(rr.str_record(rr.next()))

