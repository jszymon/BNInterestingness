from .AttrSet import Attr, AttrSet
from .ArffFileReader import read_arff_attrset

def make_arff_header(aset, quote_char = ""):
    header = ""
    relname = aset.name
    header += "@RELATION " + relname + "\n\n"
    
    for attr in aset:
        header += "@ATTRIBUTE " + attr.name
        if attr.type == "STRING":
            header += " STRING\n"
        elif attr.type == "CONTINUOUS":
            header += " REAL\n"
        elif attr.type == "CATEG":
            header += " {" + ", ".join([quote_char+str(v)+quote_char for v in attr.domain]) + "}\n"
    header += "@DATA\n"
    return header
    



if __name__ == "__main__":
    fname = "data/iris.arff"
    aset, offset = read_arff_attrset(fname)
    print(aset)
    print(offset)

    print(make_arff_header(aset))
    

