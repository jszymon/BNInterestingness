from .RecordReader import RecordReader

def _true_func(x):
    return True

class SelectionReader(RecordReader):
    """Wraps around a reader in such a way that a subset of readers
    records is read."""
    def __init__(self, reader, condition = None):
        """reader is the wrapped reader, attrs is a list of numbers of
        attributes to return."""
        if condition is None:
            condition = _true_func
        self.reader = reader
        self.condition = condition
        super(SelectionReader, self).__init__(self.reader.get_attr_set())

    def rewind(self):
        self.reader.rewind()
    def next(self):
        row = self.reader.next()
        while not self.condition(row):
            row = self.reader.next()
        return row



if __name__ == "__main__":
    from AttrSet import Attr, AttrSet
    from ListOfRecordsReader import ListOfRecordsReader
    aset = AttrSet("list", [Attr("a1", "STRING"), Attr("a2", "STRING"), Attr("a2", "STRING")])
    L = [("v1", "v2", "v3"),("txt1", "txt2", "txt3"),("abc", "def", "ghi")]
    lr = ListOfRecordsReader(L, aset)

    sr = SelectionReader(lr, lambda r: r[0][0] != 't')
    print(sr.get_attr_set())
    print()
    for r in sr:
        print(sr.str_record(r))
    print()
    sr.rewind()
    for r in sr:
        print(sr.str_record(r))
