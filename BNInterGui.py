#!/usr/bin/env python


import numpy
import random
import sys
from os.path import basename, dirname

from DataAccess import Attr, AttrSet, create_arff_reader
from BayesNet import BayesNet, read_Hugin_file, write_Hugin_file
import BayesNet.BayesNetLearn
from BayesNet import topSort
import Apriori.AprioriDistr
import time
import BayesPrune.ExactInterestingness
from BayesPrune.ExactInterestingness import BN_interestingness_exact
from BayesPrune.SamplingInterestingness import BN_interestingness_sample

from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import *


global debug
debug = 0




global minsup
global maxK
minsup = 10
maxK = 5
#minsup = 50
#maxK = 3






class PruneGUI(Frame):
    def __init__(self, master = None):
        self.bn = None
        self.ds = None
        self.bn_interestingness = None

        self.master = master
        Frame.__init__(self, self.master)
        self.create_widgets()
        self.create_BN_window()


    def create_widgets(self):
        self.pack(fill=BOTH, expand=1)
        files_grid = Frame(self)
        files_grid.pack(fill=X)

        self.data_file_name = StringVar()
        #self.data_file_name.set("data/ksl_discr.arff")
        self.data_file_name.set("")
        Label(files_grid, text="Data file").grid()
        data_file_entry = Entry(files_grid, textvariable = self.data_file_name, width = 60)
        data_file_entry.grid(row = 0, column = 1)
        data_file_button = Button(files_grid, text = "Browse", command = self.get_data_file_name)
        data_file_button.grid(row = 0, column = 2)

        self.bnet_file_name = StringVar()
        self.bnet_file_name.set("")
        Label(files_grid, text="Bayes net").grid()
        bnet_file_entry = Entry(files_grid, textvariable = self.bnet_file_name, width = 60)
        bnet_file_entry.grid(row = 1, column = 1)
        bnet_file_button = Button(files_grid, text = "Browse", command = self.get_bnet_file_name)
        bnet_file_button.grid(row = 1, column = 2)

        # control frame for algorithm parameters
        control_frame = Frame(self)
        control_frame.pack()
        self.method = StringVar()
        sample_radio = Radiobutton(control_frame, text = "sampling", variable = self.method, value = "Sampling")
        sample_radio.pack(side = LEFT)
        exact_radio = Radiobutton(control_frame, text = "exact", variable = self.method, value = "Exact")
        exact_radio.pack(side = LEFT)
        self.method.set("Sampling")
        #maxK
        self.maxK = StringVar()
        self.maxK.set("5")
        Label(control_frame, text = "  max pattern size").pack(side=LEFT)
        maxK_entry = Entry(control_frame, textvariable = self.maxK, width = 3)
        maxK_entry.pack(side=LEFT)
        #minsup
        self.minsup = StringVar()
        self.minsup.set("10")
        Label(control_frame, text = "  min inter.").pack(side=LEFT)
        minsup_entry = Entry(control_frame, textvariable = self.minsup, width = 6)
        minsup_entry.pack(side=LEFT)
        #n
        self.n = StringVar()
        self.n.set("5")
        Label(control_frame, text = "  # best patterns to find").pack(side=LEFT)
        n_entry = Entry(control_frame, textvariable = self.n, width = 4)
        n_entry.pack(side=LEFT)
        #delta
        self.delta = StringVar()
        self.delta.set("0.05")
        Label(control_frame, text = "  error prob.").pack(side=LEFT)
        delta_entry = Entry(control_frame, textvariable = self.delta, width = 5)
        delta_entry.pack(side=LEFT)

        run_button = Button(self, text = "run", command = self.run)
        run_button.pack()

        ### list of interesting attrsets
        self.selected_attrs = []
        listframe = Frame(self)
        listframe.pack(fill=BOTH, expand=1)
        yScroll = Scrollbar(listframe, orient=VERTICAL)
        self.attrset_list = Listbox(listframe, yscrollcommand=yScroll.set)
        self.attrset_list.pack(fill=BOTH, expand=1, side=LEFT)
        self.attrset_list.bind("<ButtonRelease-1>", self.attr_set_clicked)
        self.attrset_list.bind("<KeyRelease>", self.attr_set_clicked)
        yScroll.pack(side=LEFT,fill=Y)
        yScroll["command"] = self.attrset_list.yview

        # control frame for display parameters
        control_frame = Frame(self)
        control_frame.pack(side=LEFT)
        # max size
        self.maxSize = StringVar()
        self.maxSize.set("5")
        self.maxSize.trace("w", self.fill_attr_set_list)
        Label(control_frame, text = "  max shown pattern size").pack(side=LEFT)
        maxSize_entry = Entry(control_frame, textvariable = self.maxSize, width = 3)
        maxSize_entry.pack(side=LEFT)

        # max size
        self.mustHave = StringVar()
        self.mustHave.set("")
        self.mustHave.trace("w", self.must_have_attr_changed)
        Label(control_frame, text = "  must have attr").pack(side=LEFT)
        mustHave_entry = Entry(control_frame, textvariable = self.mustHave, width = 15)
        mustHave_entry.pack(side=LEFT)

        self.status_bar = Label(self, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=TOP, fill=X)

    def create_BN_window(self):
        self.bn_window = Toplevel()
        self.bn_window.title("Bayesian network")
        wframe = Frame(self.bn_window)
        wframe.pack(expand=1, fill=BOTH)

        control_frame = Frame(wframe)
        control_frame.pack(fill=X, expand=0)
        addb = Button(control_frame, text = "Add edge", command = self.add_edge)
        addb.pack(side=LEFT)
        delb = Button(control_frame, text = "Del edge", command = self.del_edge)
        delb.pack(side=LEFT)
        delallb = Button(control_frame, text = "Del all edges", command = self.del_edges)
        delallb.pack(side=LEFT)
        saveasb = Button(control_frame, text = "Save as", command = self.save_net)
        saveasb.pack(side=LEFT)
        savePSb = Button(control_frame, text = "Save as PS", command = self.save_PS)
        savePSb.pack(side=LEFT)
        self.bn_status_bar = Label(control_frame, text="", width=40, bd=1, relief=SUNKEN, anchor=W)
        self.bn_status_bar.pack(side=LEFT, fill=X, expand=1)


        bn_frame = Frame(wframe)
        bn_frame.pack(expand=1, fill=BOTH)
        bn_frame.rowconfigure(0, weight = 1)
        bn_frame.rowconfigure(1, weight = 0)
        bn_frame.columnconfigure(0, weight = 1)
        bn_frame.columnconfigure(1, weight = 0)

        xScroll = Scrollbar(bn_frame, orient=HORIZONTAL)
        xScroll.grid(row=1, column = 0, sticky=W+E)
        yScroll = Scrollbar(bn_frame, orient=VERTICAL)
        yScroll.grid(row=0, column = 1, sticky=N+S)
        self.bn_canvas = Canvas(bn_frame, xscrollcommand=xScroll.set, yscrollcommand=yScroll.set)
        self.bn_canvas.grid(row = 0, column = 0, sticky=NW+SE)
        xScroll["command"] = self.bn_canvas.xview
        yScroll["command"] = self.bn_canvas.yview

        self.click_mode = "NONE"  # how to respond to clicks
        self.click_submode = "NONE"
        self.click_state = 0
        self.bn_canvas.bind("<Button-1>", self.bn_canvas_clicked)


    def must_have_attr_changed(self, *event):
        attrnames = [n.name for n in self.bn]
        musthave_str = self.mustHave.get()
        if musthave_str not in attrnames:
            return
        self.fill_attr_set_list()


    def del_edges(self, *event):
        if self.bn is None:
            return
        if not askokcancel(message = "Are you sure you want to delete all edges?"):
            return
        BayesNet.BayesNetLearn.makeIndependentStructure(self.bn)
        self.draw_network()
        self.bn_status_bar.config(text="Edges deleted")

    def add_edge(self, *event):
        if self.bn is None:
            return
        self.bn_status_bar.config(text="Select 'from' node")
        self.click_mode = "EDGE_OP"
        self.click_submode = "ADD"
        self.click_state = 0

    def del_edge(self, *event):
        if self.bn is None:
            return
        self.bn_status_bar.config(text="Select 'from' node")
        self.click_mode = "EDGE_OP"
        self.click_submode = "DEL"
        self.click_state = 0

    def save_net(self, *event):
        if self.bn is None:
            return
        fname = asksaveasfilename(title="Save Bayesian network as...")
        of = open(fname, "w")
        write_Hugin_file(self.bn, of)
        of.close()

    def save_PS(self, *event):
        if self.bn is None:
            return
        fname = asksaveasfilename(title="Save Postscript as...",
                                  defaultextension=".eps", filetypes=[("Encaps. Postscript", "*.eps"), ("all files", "*")])
        bbox = self.bn_canvas.bbox(ALL)
        print(bbox)
        self.bn_canvas.postscript(file=fname, x = bbox[0], y = bbox[1],
                                  width = bbox[2] - bbox[0], height = bbox[3] - bbox[1])

        


    def bn_canvas_clicked(self, event):
        #print("clicked at", event.x, event.y)
        x = self.bn_canvas.canvasx(event.x)
        y = self.bn_canvas.canvasy(event.y)
        if self.bn is None:
            return
        if self.click_mode == "NONE":
            return
        cid = self.bn_canvas.find_closest(x, y)
        cid = cid[0]
        if not cid in self.id_to_node:
            return
        ni = self.id_to_node[cid]
        nname = self.bn[ni].name
        print(nname, "clicked")

        if self.click_mode == "EDGE_OP":
            if self.click_state == 0:
                self.from_node = nname
                self.click_state = 1
                self.bn_status_bar.config(text="From" + self.from_node + "; Select 'to' node")
                return
            elif self.click_state == 1:
                self.click_mode = "NONE"
                self.click_state = 0
                if nname == self.from_node:
                    showerror(message = "Same from and to nodes!")
                    self.bn_status_bar.config(text="")
                    return
                if self.click_submode == "ADD":
                    self.bn.addEdge(self.from_node, self.bn[ni].name)
                    self.bn_status_bar.config(text="Edge added")
                elif self.click_submode == "DEL":
                    self.bn.delEdge(self.from_node, self.bn[ni].name)
                    self.bn_status_bar.config(text="Edge deleted")
                else:
                    print("Wrong sub mode!")
                self.draw_network()
                return
            else:
                print("Wrong mode!")
                return

            

    def draw_network(self):
        if self.bn is None:
            return

        self.id_to_node = {} # map canvas IDs to node numbers
        self.nodenumber_to_id = {} # map canvas IDs to node numbers

        # group nodes into layers
        ts = topSort(self.bn)
        layers = []
        layer = []
        for ni in ts:
            n = self.bn[ni]
            for np in n.parents:
                if np in layer:
                    layers.append(layer)
                    layer = []
            layer.append(ni)
        layers.append(layer)

        # clear canvas
        self.bn_canvas.delete(ALL)
        # draw layers
        drawn_nodes = [None] * len(self.bn)
        y = 10
        offset = 10
        maxx = max([len(l) for l in layers])
        for l in layers:
            x = (maxx - len(l)) * 100 // 2 + offset
            for ni in l:
                drawn_nodes[ni] = (x,y)
                oval = self.bn_canvas.create_oval(x,y,x+30,y+30)
                #self.bn_canvas.itemconfig(oval, fill="red")
                self.id_to_node[oval] = ni
                self.nodenumber_to_id[ni] = oval
                txt = self.bn_canvas.create_text(x+40,y,text = self.bn[ni].name)
                self.id_to_node[txt] = ni
                for np in self.bn[ni].parents:
                    px, py = drawn_nodes[np]
                    self.bn_canvas.create_line(px+15,py+30, x+15,y, arrow=LAST)
                x += 100
            y += 70
            offset = -offset
        self.bn_canvas.config(scrollregion=self.bn_canvas.bbox(ALL))


    def attr_set_clicked(self, *event):
        """Show selected attrset on the BN"""
        if self.bn is None:
            return
        self.clear_selected_attrs()
        selindex = self.attrset_list.curselection()
        if len(selindex) == 0:
            return
        selindex = selindex[0]
        #selindex = int(selindex.strip("'"))
        aset = self.listindex_to_attrset[selindex]
        print(selindex, aset)
        self.selected_attrs = aset
        for a in self.selected_attrs:
            self.bn_canvas.itemconfig(self.nodenumber_to_id[a], fill="red")
    def clear_selected_attrs(self):
        """Unmark selected attributes on the network."""
        if self.bn is None:
            return
        for a in self.selected_attrs:
            self.bn_canvas.itemconfig(self.nodenumber_to_id[a], fill="")

    def fill_attr_set_list(self, *callparams):
        if self.bn_interestingness == None:
            return

        mode = ["attrset", "maxcell"]
        
        # read params
        maxlen_str = self.maxSize.get()
        if maxlen_str == "":
            return
        maxlen = int(maxlen_str)
        attrnames = [n.name for n in self.bn]
        musthave_str = self.mustHave.get()
        if musthave_str not in attrnames:
            must_contain_attrno = None
        else:
            must_contain_attrno = attrnames.index(musthave_str)
        
        self.clear_selected_attrs()
        self.listindex_to_attrset = {}
        self.attrset_list.delete(0, self.attrset_list.size() - 1)
        asets_selected = [x for x in self.attr_sets_w_inter if len(x[0]) <= maxlen]
        if must_contain_attrno is not None:
            asets_selected = [x for x in asets_selected if must_contain_attrno in x[0]]
        listindex = 0
        for aset, inter in [(a,i) for a, i in asets_selected]:
            if "attrset" in mode:
                self.attrset_list.insert(END, "[" + ",".join([self.bn_interestingness.ds.attrset[i].name for i in aset]) + "] " + str(inter))
                #print("[" + ",".join([self.bn_interestingness.ds.attrset[i].name for i in aset]) + "] " + str(inter))
                self.listindex_to_attrset[listindex] = aset
                listindex += 1
            if "maxcell" in mode:
                inter, distr, edistr = self.bn_interestingness.compute_attrset_interestingness(aset)
                #diff = numpy.abs(distr - edistr)
                diff = distr - edistr
                indices = numpy.ndindex(diff.shape)
                for i, d in zip(indices, numpy.ravel(diff)):
                    if abs(d) >= 0.9 * inter:
                        idomlist = [str(self.bn_interestingness.ds.attrset[a].domain[v]) for a, v in zip(aset, i)]
                        istr = ",".join(idomlist)
                        self.attrset_list.insert(END, ("    " + istr + " => " + str(d) + "  P^BN=" + str(edistr[i]) +"  P^D=" + str(distr[i])))
                        self.listindex_to_attrset[listindex] = aset
                        listindex += 1
        
        
    def run(self):
        excluded_attrs = []
        selectionCond = None

        self.status_bar.config(text="")
        self.status_bar.update()
        self.master.update()

        if self.ds is None:
            showerror(message = "No data file selected")
            return
        if self.bn is None:
            showerror(message = "No Bayesian network selected")
            return

        #print(self.bn)

        # read parameters
        maxK = int(self.maxK.get())
        minsup = int(self.minsup.get())
        ns = int(self.n.get())
        delta = float(self.delta.get())

        ### learn conditional probabilities
        self.status_bar.config(text="Learning CPTs from data")
        self.status_bar.update()
        self.master.update()

        self.ds.rewind()
        BayesNet.BayesNetLearn.learnProbabilitiesFromData(self.bn, self.ds, priorN = 0)
        self.ds.rewind()
        debug = 1
        t1 = time.time()

        ### Find interesting patterns
        self.status_bar.config(text="Computing interesting patterns...")
        self.status_bar.update()
        self.master.update()

        if self.method.get() == "Exact":
            self.bn_interestingness = BN_interestingness_exact(self.bn, self.ds)
            self.attr_sets_w_inter = self.bn_interestingness.run(minsup = minsup, maxK = maxK, apriori_debug = debug)
        elif self.method.get() == "Sampling":
            self.bn_interestingness = BN_interestingness_sample(self.bn, self.ds)
            self.attr_sets_w_inter = self.bn_interestingness.run(maxK = maxK, n = ns,
                                                                 delta = delta,
                                                                 excluded_attrs = excluded_attrs,
                                                                 selectionCond = selectionCond)
        else:
            raise RuntimeError("wrong method")

        ### print results
        t2 = time.time()
        self.status_bar.config(text="Finished!  time=" + str(t2-t1))

        self.fill_attr_set_list()
        return



    def get_data_file_name(self):
        fname = askopenfilename(title="Open .arff data file",
                                initialfile=basename(self.data_file_name.get()),
                                initialdir=dirname(self.data_file_name.get()),
                                defaultextension=".arff", filetypes=[("ARFF files", "*.arff"), ("all files", "*")])
        if fname == '' or fname == ():
            return
        try:
            self.ds = create_arff_reader(fname)
        except RuntimeError as e:
            showerror(message = "Error reading data file" + str(e))
            self.data_file_name.set("")
            self.bn = None
            return
        self.data_file_name.set(fname)
        self.bnet_file_name.set("")
        print("assuming independent structure")
        self.bn = BayesNet.BayesNet(basename(self.ds.filename), [Attr(a.name, "CATEG", a.domain) for a in self.ds.attrset])
        BayesNet.BayesNetLearn.makeIndependentStructure(self.bn)
        self.ds.rewind()
        BayesNet.BayesNetLearn.learnProbabilitiesFromData(self.bn, self.ds, priorN = 0)
        print(self.bn)
        self.draw_network()

    def get_bnet_file_name(self):
        fname = askopenfilename(title="Open .net file",
                                initialfile=basename(self.bnet_file_name.get()),
                                initialdir=dirname(self.bnet_file_name.get()),
                                defaultextension=".net", filetypes=[("NET files", "*.net"), ("all files", "*")])
        if fname == '':
            return
        try:
            self.bn = read_Hugin_file(fname)
            print(self.bn)
        except RuntimeError as e:
            showerror(message = "Could not read Bayesian network" + str(e))
            self.bnet_file_name.set("")
            self.bn = None
            return
        self.bnet_file_name.set(fname)
        self.draw_network()




if __name__ == "__main__":
    pg = PruneGUI(Tk())
    pg.mainloop()
    sys.exit(0)

