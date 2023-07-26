import DataAccess.ArffFileReader

class record_to_itemset_scanner:
    def __init__(self, fm_dataflow_scanner):
        self.ds = fm_dataflow_scanner
        aset = self.ds.get_attr_set()
        self.categ_fields = []
        self.item_numbers = []
        self.itemset_descr = []
        item_no = 0
        for i in range(len(aset)):
            a = aset[i]
            if a.type == "CATEG":
                self.categ_fields.append(a.field_no)
                self.item_numbers.append(item_no)
                for j in range(a.get_n_categories()):
                    self.itemset_descr.append((i, a.name, j))
                item_no += a.get_n_categories()
        self.n_items = item_no
    def __iter__(self):
        return self
    def next(self):
        record = self.ds.next()
        itemset = []
        for i in range(len(self.categ_fields)):
            itemset.append(self.item_numbers[i] + record[i])
        return itemset
    def rewind(self):
        self.ds.rewind()

if __name__ == "__main__":
    ds = DataAccess.ArffFileReader.create_arff_reader("../data/lenses.arff")
    ris = record_to_itemset_scanner(ds)
    for iset in ris:
        print iset
