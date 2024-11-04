from .RecordReader import RecordReader

class ListOfRecordsReader(RecordReader):
    """Class which allows for reading Python lists of records as data
    sources."""
    def __init__(self, L, attrset):
        super(ListOfRecordsReader, self).__init__(attrset)
        self.data = L
        self.rewind()

    def rewind(self):
        try:
            self.data.rewind()
        except RuntimeError:
            pass
        self.i = iter(self.data)

    def next(self):
        return self.i.next()




if __name__ == "__main__":
    from AttrSet import Attr, AttrSet
    aset = AttrSet("list", [Attr("a1", "STRING"), Attr("a2", "STRING")])
    L = [("v1", "v2"),("txt1", "txt2"),("abc", "def")]
    lr = ListOfRecordsReader(L, aset)
    for r in lr:
        print(lr.str_record(r))

