"""Reading delimited text files."""

from .AttrSet import Attr, AttrSet
from .RecordReader import RecordReader

class DelimitedFileReader(RecordReader):
    def __init__(self, attrset, datafile, offset, sep=",", strip=" \t\v", missingstr=["?"]):
        super(DelimitedFileReader, self).__init__(attrset)
        if isinstance(datafile, str):
            self.filename = datafile
            self.f = open(self.filename, "r")
        else:
            self.filename = "Unknown"
            self.f = datafile
        self.sep = sep
        self.strip = strip + "\n"
        self.missingstr = missingstr
        self.offset = offset
        self.rewind()

    def rewind(self):
        self.f.seek(self.offset)

    def __next__(self):
        line = next(self.f)#.next()
        line = line.strip()
        while len(line) > 0 and line[0] == '%':
            line = next(self.f)
            line = line.strip()
        strings = line.split(self.sep)
        row = [None] * len(self.attrset)
        for i, a in enumerate(self.attrset):
            if i >= len(strings):
                #val = DataAccess.missing()
                val = None
            else:
                strval = strings[i].strip(self.strip)
                if a.type == "CATEG":
                    if strval in a.domain:
                        val = a.domain.index(strval)
                    elif strval in self.missingstr:
                        #val = DataAccess.missing()
                        val = None
                    else:
                        raise RuntimeError("value "+ strval +" not in domain")
                elif a.type == "CONTINUOUS":
                    try:
                        val = float(strval)
                    except ValueError:
                        if strval in self.missingstr:
                            #val = DataAccess.missing()
                            val = None
                        else:
                            raise RuntimeError("value not a number")
                elif a.type == "STRING":
                    val = strval
                else:
                    raise RuntimeError("Wrong attribute type")
            row[i] = val
        return row


if __name__ == "__main__":
    aset = AttrSet("iris")
    aset.append_attr(Attr("sepallength", "CONTINUOUS"))
    aset.append_attr(Attr("sepalwidth", "CONTINUOUS"))
    aset.append_attr(Attr("petallength", "CONTINUOUS"))
    aset.append_attr(Attr("petalwidth", "CONTINUOUS"))
    aset.append_attr(Attr("class", "CATEG", domain = ["Iris-setosa","Iris-versicolor","Iris-virginica"]))
    print(aset)
    dfr = DelimitedFileReader(aset, "data/iris.arff", 2930)
    for x in dfr:
        print(dfr.str_record(x))
