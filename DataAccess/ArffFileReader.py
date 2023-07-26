from .AttrSet import Attr, AttrSet
from .DelimitedFileReader import DelimitedFileReader


def read_arff_attrset(arfffile):
    if isinstance(arfffile, str):
        f = open(arfffile, "r")
    else:
        f = arfffile
    relname = ""
    aset = AttrSet()
    rawline = " "
    while rawline:
        rawline = f.readline()
        line = rawline.strip()
        if len(line) == 0:
            continue
        if line[0] == '%':
            continue
        strings = line.split()
        if strings[0].lower() == "@relation":
            if len(strings) >= 2:
                relname = strings[1]
        elif strings[0].lower() == "@attribute":
            if len(strings) < 3:
                raise RuntimeError("Missing attribute name or type")
            if strings[2].lower() in ["real", "continuous", "integer", "numeric"]:
                atype = "CONTINUOUS"
                domain = []
            elif rawline.find("{") != -1:
                atype = "CATEG"
                b = rawline.find("{")
                e = rawline.find("}")
                if e == -1 or b > e:
                    raise RuntimeError("syntax error in domain")
                domstr = (rawline[b+1:e]).strip()
                domain = [v.strip().strip("'") for v in domstr.split(",")]
            else:
                raise RuntimeError("Wrong attribute type or missing domain")
            aset.append_attr(Attr(strings[1], atype, domain))
        elif strings[0].lower() == "@data":
            break
    offset = f.tell()
    aset.name = relname
    return aset, offset

def create_arff_reader(arfffile):
    aset, offset = read_arff_attrset(arfffile)
    dfr = DelimitedFileReader(aset, arfffile, offset, strip="'")
    dfr.relname = aset.name
    return dfr

if __name__ == "__main__":
    fname = "data/iris.arff"
    aset, offset = read_arff_attrset(fname)
    ar = create_arff_reader(fname)
    for row in ar:
        print(ar.str_record(row))
    ar = create_arff_reader("data/lenses.arff")
    for row in ar:
        print(ar.str_record(row))
    ar = create_arff_reader("data/lenses2.arff")
    for row in ar:
        print(ar.str_record(row))
