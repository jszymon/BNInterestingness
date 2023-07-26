from .AttrSet import Attr, AttrSet
from .RecordReader import RecordReader

class ProjectionReader(RecordReader):
    """Wraps around a reader in such a way that a subset of reader's
    attributes is read."""
    def __init__(self, reader, attr_numbers):
        """reader is the wrapped reader, attrs is a list of numbers of
        attributes to return."""
        self.reader = reader
        self.attr_numbers = attr_numbers
        asetname = self.reader.get_attr_set().name + "[" + ",".join([self.reader.get_attr_set()[i].name for i in self.attr_numbers]) + "]"
        aset = AttrSet(asetname, [self.reader.get_attr_set()[i] for i in self.attr_numbers])
        super(ProjectionReader, self).__init__(aset)

    def rewind(self):
        self.reader.rewind()
    def __next__(self):
        row = self.reader.next()
        newrow = [row[i] for i in self.attr_numbers]
        return newrow



if __name__ == "__main__":
    from ListOfRecordsReader import ListOfRecordsReader
    aset = AttrSet("list", [Attr("a1", "STRING"), Attr("a2", "STRING"), Attr("a2", "STRING")])
    L = [("v1", "v2", "v3"),("txt1", "txt2", "txt3"),("abc", "def", "ghi")]
    lr = ListOfRecordsReader(L, aset)

    pr = ProjectionReader(lr, [0,2])
    print(pr.get_attr_set())
    print()
    for r in pr:
        print(pr.str_record(r))
    print()
    pr.rewind()
    for r in pr:
        print(pr.str_record(r))

