import shlex
import numpy

from DataAccess import Attr
from BayesNet import BayesNet
from .NoisyOR import NoisyOR


class BNReadError(RuntimeError):
    pass

def check_token(string, tok, lex):
    """Assert that tok==string."""
    if tok != string:
        msg = lex.error_leader() + "'" + string + "' expected, got '" + tok + "' instead"
        raise RuntimeError(msg)
    
def match_token(string, lex):
    """Get given string from the lexer."""
    tok = lex.get_token()
    check_token(string, tok, lex)

def parse_list(lex):
    lst = []
    tok = lex.get_token()
    while tok != ")" and tok != "":
        lex.push_token(tok)
        val = parse_value(lex)
        lst.append(val)
        tok = lex.get_token()
    check_token(")", tok, lex)
    #lex.push_token(tok)
    #match_token(")", lex)
    return lst
    
def parse_value(lex):
    tok = lex.get_token()
    #print tok, len(tok), tok == "\""
    if len(tok) == 0:
        msg = lex.error_leader() + "Empty token"
        raise RuntimeError(msg)
    if tok[0] == "\"":
        val = tok.strip("\"")
        if not tok.endswith("\""):
            msg = lex.error_leader() + "String token not ending with '\"'"
            raise RuntimeError(msg)
    elif tok == "(":
        val = parse_list(lex)
    else:  # should be a Hugin expression (currently a number or NoisyOR)
        if tok == "NoisyOR":
            val = ["NoisyOR"]
            tok2 = lex.get_token()
            check_token("(", tok2, lex)
            while tok2 != ")":
                name = lex.get_token()
                comma = lex.get_token()
                check_token(",", comma, lex)
                tok2 = lex.get_token()
                try:
                    prob = float(tok2)
                except ValueError:
                    msg = lex.error_leader() + "Wrong probability value '" + tok2 + "'"
                    raise RuntimeError(msg)
                tok2 = lex.get_token()
                if tok2 != ")" and tok2 != ",":
                    msg = lex.error_leader() + "Expected ')' or ',', got '" + tok2 + "'"
                    raise RuntimeError(msg)
                val.append((name, prob))
        else: # should be a number
            try:
                val = float(tok)
            except ValueError:
                msg = lex.error_leader() + "Wrong number string '" + tok + "'"
                raise RuntimeError(msg)
    return val

def parse_node(node, lex):
    match_token("{", lex)
    tok = lex.get_token()
    while tok != "}" and tok != "":
        param = tok
        match_token("=", lex)
        value = parse_value(lex)
        node.params[param] = value
        match_token(";", lex)
        tok = lex.get_token()
    check_token("}", tok, lex)


def parse_attrs(lex):
    """Parse attributes of a potential."""
    match_token("(", lex)
    targets = []
    while True:
        tok = lex.get_token()
        if tok in ["|", ")", "}"]:
            break
        targets.append(tok)
    conditioned_on = []
    if tok == "|":
        tok = lex.get_token()
        while tok != ")" and tok != "":
            conditioned_on.append(tok)
            tok = lex.get_token()
    check_token(")", tok, lex)
    return (targets, conditioned_on)

class HuginNode(object):
    def __init__(self, _class, name, node_type = "discrete"):
        self._class = _class
        self.node_type = node_type
        self.name = name
        self.params = {}
        self.targets = None
        self.conditioned_on = None
    def __str__(self):
        ret = ""
        ret += self._class + " " + self.node_type + " " + self.name
        if self.targets is not None:
            ret += "(%s | %s)" % (self.targets, str(self.conditioned_on))
        ret += "\n" + str(self.params)
        return ret
    def __repr__(self):
        return self.__str__()

def parse_net(net, lex):
    tok = lex.get_token()
    while tok != "":
        nodetype = "discrete"
        if tok in ["discrete", "continuous"]:
            nodetype = tok
            tok = lex.get_token()
        nodeclass = tok
        if nodeclass == "potential":
            attrs = parse_attrs(lex)
        tok = lex.get_token()
        if tok == "{":
            nodename = ""
            lex.push_token(tok)
        else:
            nodename = tok
        node = HuginNode(nodeclass, nodename, nodetype)
        if nodeclass == "potential":
            node.targets = attrs[0]
            node.conditioned_on = attrs[1]
        parse_node(node, lex)
        net.append(node)
        tok = lex.get_token()    

def parse_distr(potential, shape):
    """Parse potential's distribution."""
    if 'data' in potential.params:
        distr = numpy.array(potential.params['data'])
        distr.shape = shape
    elif 'model_data' in potential.params:
        nor = NoisyOR(potential.conditioned_on, shape[:-1])
        for vname, prb in potential.params['model_data'][0][1:]:
            nor.add_variable(vname, prb)
        distr = nor.get_table()
    else:
        distr = None
    return distr
    
def read_Hugin_file(file_name):
    """Return a BayesNet read from a Hugin .net file.

    Files ending with .gz are treated as gzip compressed."""
    # parse the file
    try:
        if file_name.endswith(".gz"):
            import gzip
            file = gzip.GzipFile(file_name)
        else:
            file = open(file_name, "r")
    except IOError as ie:
        raise BNReadError("Error reading bayesian network: " + ie.strerror + ":" + ie.filename)
    lex = shlex.shlex(file, file_name)
    lex.commenters = "%"
    lex.wordchars += "-+."
    lex.quotes = "\""
    net = []
    parse_net(net, lex)

    ### build the network
    # get nodes
    attrs = []
    domains = {}
    ids_2_names = {} # map node id's to node labels
    ids_2_numbers = {}
    for n in net:
        if n._class == "node": 
            if n.node_type != "discrete":
                raise RuntimeError("Only discrete nodes are supported")
            name = n.params.get('label', n.name)
            if n.name in ids_2_names:
                raise RuntimeError("Repeated definition for node " + n.name)
            ids_2_names[n.name] = name
            ids_2_numbers[n.name] = len(attrs)
            attrs.append(Attr(name, "CATEG", n.params['states']))
            domains[n.name] = n.params['states']
    bn = BayesNet(file_name, attrs)
    # get potentials
    for p in net:
        if p._class == "potential":
            shape = []
            for a in p.conditioned_on + p.targets:
                shape.append(len(domains[a]))
            distr = parse_distr(p, shape)
            if distr is None:
                raise RuntimeError("No distribution for node " + str(ids_2_names[p.targets[0]]))
            targets = [ids_2_numbers[c] for c in p.targets]
            conditioned_on = [ids_2_numbers[c] for c in p.conditioned_on]
            if len(targets) == 1:  # ordinary node
                bn[targets[0]].set_parents_distr(conditioned_on, distr)
            else:
                # joint node
                if len(conditioned_on) > 0:
                    raise RuntimeError("Joint nodes cannot have parents.")
                jn = bn.addJointDistr(targets)
                jn.distr = distr
                print("Warning!!!: joint potential distributions read as dense.")
    return bn


def write_Hugin_file(bn, f):
    """Write a Bayesian network bn to a file f in Hugin format."""
    print("""net
{
    node_size = (80 40);
    HR_Grid_X = "10";
    HR_Grid_Y = "10";
    HR_Grid_GridSnap = "1";
    HR_Grid_GridShow = "0";
    HR_Font_Name = "Arial";
    HR_Font_Size = "-12";
    HR_Font_Weight = "400";
    HR_Font_Italic = "0";
    HR_Propagate_Auto = "1";
}""", file=f)
    for i, n in enumerate(bn):
        print("\nnode V_" + str(i), file=f)
        print("{", file=f)
        print("  states = (" + " ".join(["\""+str(x)+"\"" for x in n.domain]) + ");", file=f)
        print("  label = \"" + n.name + "\";", file=f)
        print("  position = (0 0);", file=f)
        print("}", file=f)
    for i, n in enumerate(bn):
        if not n.in_joint:
            if len(n.parents) > 0:
                cond = "| " + " ".join(["V_"+str(x) for x in n.parents])
            else:
                cond = ""
            print("\npotential ( V_" + str(i) + cond + ")", file=f)
            print("{", file=f)
            print("  data = (" + " ".join([str(x) for x in n.distr.flat]) + ");", file=f)
            print("}", file=f)
    for i, jn in enumerate(bn.joint_distrs):
        vars_str = " ".join("V_"+str(x) for x in jn.nodes)
        print("\npotential ( " + vars_str + ")", file=f)
        print("{", file=f)
        print("  data = (" + " ".join([str(x) for x in jn.distr.to_array().flat]) + ");", file=f)
        print("}", file=f)


def main():
    #bn = read_Hugin_file("oil.net")
    bn = read_Hugin_file("../BayesPrune/data/ksl_discr.net")
    #bn = read_Hugin_file("ksl.net")
    #bn = read_Hugin_file("/home/szymon/dmining/Python/BayesPrune/data/soybean.net")
    #bn = read_Hugin_file("/home/szymon/dmining/data/Borreliosis/borrelia200603.net")
    print(bn)
    #raise BNReadError, "txt"
    f = open("/tmp/bn", "w")
    write_Hugin_file(bn, f)

if __name__ == "__main__":
    main()
