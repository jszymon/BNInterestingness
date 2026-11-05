#!/usr/bin/env python

import numpy as np

from ..DataAccess import Attr, AttrSet
from .BNutils import blockiter, distr_2_str
from ..Utils import SparseDistr
from .NoisyOR import NoisyOR
    



class BayesNode(Attr):
    def __init__(self, bnet, attr):
        """Create Bayesian network node.

        Initially the node has no parents and a uniform distribution."""
        super(BayesNode, self).__init__(attr.name, attr.type, attr.domain)
        self.bnet = bnet # reference to BayesNet the node is in
        self.parents = []
        self.nd = len(self.domain)
        self.in_joint = False  # whether the node is member of a joint distribution
        self.distr = np.full(self.nd, + 1.0/self.nd)
        # None for nodes given by a conditional probability table.  For
        # Noisy-OR nodes a dict {'leak': p, 'inhibit': {parent_number: p}}.
        self.noisy_or = None
        # None, or a dict {'C': C, 'standardize': bool} for nodes whose
        # table is given by a logistic regression learned from data.
        self.logistic = None

    def is_noisy_or(self):
        return self.noisy_or is not None

    def is_logistic(self):
        return self.logistic is not None

    def distr_type(self):
        """Return 'CPT', 'NoisyOR' or 'Logistic'."""
        if self.noisy_or is not None:
            return 'NoisyOR'
        if self.logistic is not None:
            return 'Logistic'
        return 'CPT'

    def set_logistic(self, C = float("inf"), standardize = True):
        """Make the node's table a logistic regression of its parents.

        The table is (re)computed when probabilities are learned from
        data.  C is the inverse regularization strength as in
        scikit-learn (inf: no regularization); standardize says whether
        the (one-hot encoded) parents are standardized."""
        if self.in_joint:
            raise ValueError(f"Node {self.name} is part of a joint distribution.")
        C = float(C)
        if not C > 0:
            raise ValueError(f"C must be positive (inf for no regularization), got {C}.")
        self.noisy_or = None
        self.logistic = {'C': C, 'standardize': bool(standardize)}

    def set_noisy_or(self, leak = 1.0, inhibit = None):
        """Make the node a Noisy-OR gate of its current parents.

        The last value of the node and of each parent means ON, all
        other values mean OFF; the node must be binary.  leak is the
        inhibition probability of an always-ON cause, i.e. the
        probability that the node is OFF when all parents are OFF.
        inhibit maps parents (names or numbers) to the probability that
        the parent being ON fails to turn the node ON.  Parents missing
        from inhibit get 1.0 (no influence)."""
        if self.nd != 2:
            raise ValueError(f"Noisy-OR node {self.name} must have exactly 2 values, has {self.nd}.")
        if self.in_joint:
            raise ValueError(f"Node {self.name} is part of a joint distribution.")
        inh = {}
        for k, p in (inhibit or {}).items():
            if isinstance(k, str):
                k = self.bnet.names_to_numbers([k])[0]
            if k not in self.parents:
                raise ValueError(f"{self.bnet[k].name} is not a parent of {self.name}.")
            inh[k] = float(p)
        for p in [leak] + list(inh.values()):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Probability {p} not in [0, 1].")
        self.logistic = None
        self.noisy_or = {'leak': float(leak), 'inhibit': inh}
        self.update_noisy_or_distr()

    def update_noisy_or_distr(self):
        """Recompute the probability table of a Noisy-OR node."""
        shape = [self.bnet[i].nd for i in self.parents]
        nor = NoisyOR(list(self.parents), shape)
        nor.add_variable("true", self.noisy_or['leak'])
        for i, p in self.noisy_or['inhibit'].items():
            nor.add_variable(i, p)
        self.distr = nor.get_table()

    def set_cpt(self):
        """Make the node use a plain conditional probability table
        (the current table is kept)."""
        self.noisy_or = None
        self.logistic = None

    def fix_data(self):
        """Fix node params which are known only after the full network is created"""

    def P(self, x):
        """Get (conditional) probability for given vector x."""
        idx = [x[i] for i in self.parents]
        return self.distr[tuple(idx)]
    def normalizeProbabilities(self):
        """Normalize all conditional probabilities in a node so that
        they add up to 1.0."""
        ri = len(self.domain)
        i = 0
        flat = np.ravel(self.distr)
        while i < len(flat):
            s = sum(flat[i:i+ri])
            for j in range(i, i+ri):
                flat[j] /= s
            i += ri

    def set_parents_distr(self, parents, distr):
        if len(parents) > 0 and isinstance(parents[0], str):
            parents = self.bnet.names_to_numbers(parents)
        self.parents = parents
        # TODO: verify distr size
        self.distr = distr
        self.noisy_or = None
        self.logistic = None

    def del_all_parents(self):
        self.parents = []
        self.noisy_or = None
        self.logistic = None
        nd = len(self.domain)
        self.distr = np.zeros(nd) + 1.0/nd

    def __str__(self):
        ret = super(BayesNode, self).__str__()
        ret += "\nParents: " + str([self.bnet[i].name for i in self.parents])
        if self.in_joint:
            ret += "\nPart of a joint distribution\n"
        else:
            if self.noisy_or is not None:
                inh = {self.bnet[i].name: p for i, p in self.noisy_or['inhibit'].items()}
                ret += f"\nNoisy-OR: leak={self.noisy_or['leak']:g} inhibit={inh}"
            if self.logistic is not None:
                ret += f"\nLogistic regression: C={self.logistic['C']:g} standardize={self.logistic['standardize']}"
            ret += "\nDistribution:\n" + distr_2_str(self.distr, cond = True)
        return ret


class JointNode:
    def __init__(self, bnet, nodes, distr=None):
        """Create a node representing a joint distribution of several variables.

        The variables are also present as ordinary nodes.  Nodes
        present in JointNode cannot have parents.

        """
        self.bnet = bnet
        for i in nodes:
            if not isinstance(i, int):
                raise RuntimeError(f"Nodes in a joint must be given as ints.  Got {i}")
        self.nodes = nodes
        self.shape = tuple(self.bnet[i].nd for i in self.nodes)
        if distr is None:
            distr = SparseDistr(self.shape)
        self.distr = distr
    def P(self, x):
        """Get (conditional) probability for given vector x."""
        idx = [x[i] for i in self.nodes]
        return self.distr[tuple(idx)]

    def __str__(self):
        ret = "JointNode: " + str([self.bnet[i].name for i in self.nodes])
        ret += "\nDistribution:\n" + str(self.distr)
        return ret

class BayesNet(AttrSet):
    def __init__(self, name = "", attrs = None):
        nodes = [BayesNode(self, a) for a in attrs]
        super(BayesNet, self).__init__(name, nodes)
        self.joint_distrs = []


    def addEdge(self, src_name, dst_name):
        """Add an edge from attribute src_name to attribute dst_name.""" 
        i = self.names_to_numbers([src_name])[0]
        src_node = self[src_name]
        dst_node = self[dst_name]
        if i in dst_node.parents:
            raise RuntimeError("Nodes already connected")
        # TODO: check if cycles aren't introduced
        dst_node.distr = np.array([dst_node.distr] * len(src_node.domain))
        dst_node.parents.insert(0,i)
        if dst_node.is_noisy_or():
            dst_node.update_noisy_or_distr()

    def delEdge(self, src_name, dst_name):
        """Delete the edge from attribute src_name to attribute dst_name.""" 
        i = self.names_to_numbers([src_name])[0]
        dst_node = self[dst_name]
        if i not in dst_node.parents:
            raise RuntimeError("Edge does not exist")
        dst_node.distr = np.sum(dst_node.distr, dst_node.parents.index(i)) / 2
        del dst_node.parents[dst_node.parents.index(i)]
        if dst_node.is_noisy_or():
            dst_node.noisy_or['inhibit'].pop(i, None)
            dst_node.update_noisy_or_distr()

    def addJointDistr(self, nodes, distr=None):
        """Declares that given nodes will be modeled using their joint
        distribution.

        This turns the network into a chain graph.  Currently nodes in
        the group cannot have parents.  Their current parents will be
        removed.  Joint distrubutions cannot overlap (an error will be
        raised).

        """
        if len(nodes) > 0 and isinstance(nodes[0], str):
            ni = self.names_to_numbers(nodes)
        else:
            ni = nodes
        for i in ni:
            name = self.numbers_to_names([i])[0]
            # check that nodes are not already in a joint distribution.
            for j in self.joint_distrs:
                if i in j.nodes or self[i].in_joint:
                    raise RuntimeError(f"Node {name} already part of a joint distribution.")
            # check that nodes have no patents
            if len(self[i].parents) != 0:
                raise RuntimeError(f"Nodes in a joint distribution cannot have parents (node {name}).")
        for i in ni:
            self[i].in_joint = True
            self[i].noisy_or = None
            self[i].logistic = None
        jn = JointNode(self, ni, distr)
        self.joint_distrs.append(jn)
        return jn
    def delJointDistr(self, node):
        """Deletes the joint distribution node is in.

        Returns list of numbers of nodes which were in the joint.

        """
        if isinstance(node, str):
            node = self.names_to_numbers([node])[0]
        if not self[node].in_joint:
            raise RuntimeError(f"Node {self[node].name} not in a joint distribution.")
        # find joint node
        for i, jn in enumerate(self.joint_distrs):
            if node in jn.nodes:
                break
        else:
            raise RuntimeError(f"BN inconstitent: {self[node].name} has .in_joint=True but is not in any joint distribution.")
        for nj in jn.nodes:
            self[nj].in_joint = False
        self.joint_distrs.pop(i)
        return jn.nodes

    def validate(self, err = 0.00001):
        """Validates the network.

        Currently only checks that each distribution is a conditional
        distribution."""
        # TODO: acyclicity, distr sizes
        for n in self:
            ri = len(n.domain)
            for subdistr in blockiter(np.ravel(n.distr), ri):
                if abs(sum(subdistr) - 1.0) > err:
                    msg=f"Conditional distribution for {n.name} does not sum to 1.0 for all parent combinations"
                    raise ValueError(msg)
                    #print("error in node", n.name, "prob sum=", sum(subdistr))


    def normalizeProbabilities(self):
        """Normalize all conditional probabilities in all nodes so
        that they add up to 1.0."""
        for n in self:
            if not n.in_joint:
                n.normalizeProbabilities()
        for jn in self.joint_distrs:
            jn.distr.normalize()

    def P(self, x):
        """Return the probability of input vector x"""
        P = 1.0
        for i, n in enumerate(self):
            if not n.in_joint:
                P *= n.P(x)[x[i]]
        for jn in self.joint_distrs:
            P *= jn.P(x)
        return P

    def get_shape(self):
        """Get the shape of the joint distribution."""
        shape = [len(a.domain) for a in self]
        return shape

    def jointP(self):
        """Return the numpy for the joint distribution of the network."""
        shape = tuple(self.get_shape())
        d = np.zeros(shape)
        for x in np.ndindex(shape):
            d[x] = self.P(x)
        return d
    
    def __str__(self):
        ret = ""
        ret += "Bayesian network: " + self.name
        ret += "\nNodes:\n"
        ret += "\n".join([str(node) for node in self])
        if len(self.joint_distrs) > 0:
            ret += "\nJoint distributions:\n"
            ret += "\n".join([str(jn) for jn in self.joint_distrs])
            ret += "\n"
        return ret

