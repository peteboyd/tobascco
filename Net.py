import math
import sys
from sage.all import *
from uuid import uuid4
import numpy as np
from scipy.optimize import fmin, minimize 

class SystreDB(dict):
    """A dictionary which reads a file of the same format read by Systre"""
    def __init__(self, filename=None):
        self.voltages = {}
        self.read_store_file(filename)

    def read_store_file(self, file=None):
        """Reads and stores the nets in the self.file file.
        Note, this is specific to a systre.arc file and may be subject to
        change in the future depending on the developments ODF makes on
        Systre.

        """
        # just start an empty list 
        if file is None:
            return

        f = open(file, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            l = line.strip().split()
            if l and l[0] == 'key':
                k = l[1]
                e = list(self.Nd_chunks([int(i) for i in l[2:]], 3))
                # generate random unique uuid name for the dictionary
                # incase the 'name' column isn't found
                name = self.read_chunk(f)
                g, v = self.gen_sage_graph_format(e)
                self[name] = g
                self.voltages[name] = np.matrix(v)

    def read_chunk(self, fileobject):
        name = uuid4()
        for j in range(6):
            r = fileobject.readline()
            xline = r.strip().split()
            if xline[0] == 'id':
                name = xline[1]
        return name

    def Nd_chunks(self, list, dim):
        n = 2+dim
        for i in xrange(0, len(list), n):
            yield tuple(list[i:i+n])

    def gen_sage_graph_format(self, edges):
        """Take the edges from a systre db file and convert 
        to sage graph readable format.
        
        Assumes that the direction of the edge goes from
        [node1] ---> [node2]
        """
        sage_dict = {}
        voltages = []
        for id, (v1, v2, e1, e2, e3) in enumerate(edges):
            ename = 'e%i'%(id+1)
            voltages.append((e1, e2, e3))
            try:
                n1 = chr(v1-1 + ord("A"))
            except ValueError:
                n1 = str(v1-1)
            try:
                n2 = chr(v2-1 + ord("A"))
            except ValueError:
                n2 = str(v2-1)

            sage_dict.setdefault(n1, {})
            sage_dict.setdefault(n2, {})
            sage_dict[n1].setdefault(n2, [])
            sage_dict[n1][n2].append(ename)
        return (sage_dict, voltages)

class Net(object):

    def __init__(self, graph=None, dim=3):
        self.lattice_basis = None
        self.metric_tensor = None
        self.cycle = None
        self.cycle_rep = None
        self.cocycle = None
        self.cocycle_rep = None
        self.periodic_rep = None # alpha(B)
        self.edge_labels = None
        self.node_labels = None
        self.colattice_dotmatrix = None
        self.voltage = None
        self._graph = graph
        # n-dimensional representation, default is 3
        self.ndim = dim
        if graph is not None:
            self._graph = DiGraph(graph, multiedges=True, loops=True)

    def get_cocycle_basis(self):
        """The orientation is important here!"""
        size = self._graph.order() - 1
        len = self._graph.size()
        count = 0
        for vert in self._graph.vertices():
            if count == size:
                break
            vect = np.zeros(len)
            out_edges = self._graph.outgoing_edges(vert)
            inds = self.return_indices(out_edges)
            if inds:
                vect[inds] = 1.
            in_edges = self._graph.incoming_edges(vert)
            inds = self.return_indices(in_edges)
            if inds:
                vect[inds] = -1.
            if self.cycle_cocycle_check(vect):
                count += 1
                v = np.reshape(vect, (1,self.shape))
                if self.cocycle is None:
                    self.cocycle = v 
                else:
                    self.cocycle = np.concatenate((self.cocycle,v))

        if count != size:
            print "ERROR - could not find a linear independent cocycle basis!"
            sys.exit()
        self.cocycle = np.matrix(self.cocycle)
        self.cocycle_rep = np.matrix(np.zeros((size, self.ndim)))
  
    def cycle_cocycle_check(self, vect):
        if self.cocycle is None and self.cycle is None:
            return True
        elif self.cocycle is None and self.cycle is not None:
            return self.check_linear_dependency(vect, self.cycle)
        else:
            return self.check_linear_dependency(vect, self.cycle_cocycle)

    def get_cycle_basis(self):
        """Find the basis for the cycle vectors. The total number of cycle vectors
        in the basis is E - V + 1 (see n below). Once this number of cycle vectors is found,
        the program returns.

        NB: Currently the cycle vectors associated with the lattice basis are included
        in the cycle basis - this is so that the embedding of the barycentric placement
        of the net works out properly. Thus the function self.get_lattice_basis() 
        should be called prior to this.
        
        """

        c = self.iter_cycles(node=self._graph.vertices()[0],
                             used=[],
                             nodes_visited=[],
                             cycle_baggage=[],
                             counter=0)
        n = self.shape - self.num_nodes + 1
        count = 0
        if self.lattice_basis is not None:
            self.cycle = self.add_to_matrix(self.lattice_basis, self.cycle)
            self.cycle_rep = self.add_to_matrix(np.identity(self.ndim), self.cycle_rep)

        for id, cycle in enumerate(c):
            if count >= n:
                break
            vect = np.zeros(self.shape)
            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            volt = self.get_voltage(vect)
            # REPLACE WITH CHECK_LINEAR_DEPENDENCY()
            check = self.cycle_cocycle_check(vect)
            if np.all(np.abs(volt) < 1.001) and check:
                self.add_to_matrix(vect, self.cycle)
                self.add_to_matrix(volt, self.cycle_rep)
                count += 1
        self.cycle = np.matrix(self.cycle)
        self.cycle_rep = np.matrix(self.cycle_rep)
        del c

    def add_to_matrix(self, vect, rep):
        """Works assuming the dimensions are the same"""
        if len(vect.shape) == 1:
            v = np.reshape(vect,(1,vect.shape[-1]))
        else:
            v = vect
        if rep is None:
            return v.copy()
        else:
            return np.concatenate((rep, v))

    def get_voltage(self, cycle):
        return np.array(cycle*self.voltage)[0]

    def debug_print(self, val, msg):
        print "%s[%d] %s"%("  "*val, val, msg)

    def iter_cycles(self, node=None, edge=None, cycle=[], used=[], nodes_visited=[], cycle_baggage=[], counter=0):
        """Recursive method to iterate over all cycles of a graph.
        NB: Not tested to ensure completeness, however it does find cycles.
        NB: Likely produces duplicate cycles along different starting points
        **last point fixed but not tested**

        """
        if node is None:
            node = self.graph.vertices()[0]

        if node in nodes_visited:
            i = nodes_visited.index(node)
            nodes_visited.append(node)
            cycle.append(edge)
            used.append(edge[:3])
            c = cycle[i:]
            uc = sorted([j[:3] for j in c])

            if uc in cycle_baggage:
                pass
            else:
                cycle_baggage.append(uc)
                yield c
        else:
            nodes_visited.append(node)
            if edge:
                cycle.append(edge)
                used.append(edge[:3])
            e = [(x, y, z, 1) for x, y, z in
                    self.graph.outgoing_edges(node)
                    if (x,y,z) not in used]
            e += [(x, y, z, -1) for x, y, z in
                    self.graph.incoming_edges(node)
                    if (x,y,z) not in used]
            for j in e:
                newnode = j[0] if j[0]!=node else j[1]
                #msg = "test: (%s to %s) via %s"%(node, newnode, j[2])
                #self.debug_print(counter, msg)
                for val in self.iter_cycles( 
                                   node=newnode,
                                   edge=j,
                                   cycle=cycle, 
                                   used=used, 
                                   nodes_visited=nodes_visited,
                                   cycle_baggage=cycle_baggage,
                                   counter=counter+1):
                    yield val
                nodes_visited.pop(-1)
                cycle.pop(-1)
                used.pop(-1)

    def get_edges_from_index(self, cycle):
        edge_names = ['e%i'%(i+1) for i in np.nonzero(cycle)[0]]
        ret = []
        for n in edge_names:
            ed = None
            for edge in self._graph.edges():
                if edge[2] == n:
                    ed = edge
                    break
            ret.append(ed)
        return ret

    def get_lattice_basis(self):
        """Obtains a lattice basis by iterating over all the cycles and finding
        ones with net voltages satisifying one of the n dimensional basis vectors.

        """
        basis_vectors = []
        c = self.iter_cycles(node=self._graph.vertices()[0],
                             used=[],
                             nodes_visited=[],
                             cycle_baggage=[],
                             counter=0)
        self.lattice_basis = np.zeros((self.ndim, self.shape))
        for cycle in c:
            vect = np.zeros(self.shape)
            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            volt = self.get_voltage(vect)
            for id, e in enumerate(np.identity(self.ndim)):
                if np.allclose(np.abs(volt), e):
                    check = self.check_linear_dependency(vect, self.lattice_basis[basis_vectors])
                    if id not in basis_vectors and check:
                        basis_vectors.append(id)
                        self.lattice_basis[id] = volt[id]*vect
                    elif (np.count_nonzero(vect) < np.count_nonzero(self.lattice_basis[id])) and check:
                        self.lattice_basis[id] = volt[id]*vect
        if len(basis_vectors) != self.ndim:
            print "ERROR: could not find all cycle vectors for the lattice basis!"
            sys.exit()

    def check_linear_dependency(self, vect, set):
        if not np.any(set):
            return True
        else:
            A = np.concatenate((set, np.reshape(vect, (1, self.shape))))
        U, s, V = np.linalg.svd(A)
        if np.all(s > 0.0001):
            return True
        return False

    def get_index(self, edge):
        return int(edge[2][1:])-1

    def return_indices(self, edges):
        return [self.get_index(i) for i in edges]

    def return_coeff(self, edges):
        assert edges[0][3]
        return [i[3] for i in edges]

    def get_arcs(self):
        cocycle_rep = np.zeros((self.cocycle.shape[0], self.ndim))
        return self.cycle_cocycle.I*np.concatenate((self.cycle_rep, cocycle_rep),
                                                    axis=0)

    def get_2d_params(self):
        self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        lena=math.sqrt(self.metric_tensor[0,0])
        lenb=math.sqrt(self.metric_tensor[1,1])
        gamma=math.acos(self.metric_tensor[1,0]/lena/lenb)*360/(2*math.pi)
        return lena, lenb, gamma

    def min_function(self, cocyc_proj):
        cocyc_rep = np.reshape(cocyc_proj, (self.cocycle.shape[0], 3))
        self.periodic_rep = np.concatenate((self.cycle_rep,
                                            cocyc_rep),
                                           axis = 0)
        M = self._lattice_arcs*\
                (self.lattice_basis*self.projection*self.lattice_basis.T)\
                *self._lattice_arcs.T
         
        nz = np.nonzero(self.cocycle.T*self.cocycle)
        return np.sum(np.absolute(M[nz] - self.colattice_dotmatrix[nz]))

    def get_embedding(self, init_guess=None):
        if init_guess is None:
            init_guess = (np.zeros((self.cocycle.shape[0], 3))).flatten()
        #cocycle = fmin(self.min_function, init_guess, maxfun=1e5, 
        #                xtol=0.0000001, ftol=0.0000001)

        bounds = [(-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1)]
        cocycle = minimize(self.min_function, init_guess, method="L-BFGS-B", 
                            tol=0.0000001, bounds=bounds)
        return np.concatenate((self.cycle_rep, 
                        np.reshape(cocycle.x, (self.cocycle.shape[0],3))),
                        axis=0)

    def get_metric_tensor(self):
        self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        #self.metric_tensor = self.lattice_basis*self.eon_projection*self.lattice_basis.T

    def barycentric_embedding(self):
        self.cocycle_rep = np.zeros((self.cocycle.shape[0], 3))
        self.periodic_rep = np.concatenate((self.cycle_rep,
                                            self.cocycle_rep),
                                            axis = 0)
        self.get_metric_tensor()
        

    def get_3d_params(self):
        lena = math.sqrt(self.metric_tensor[0,0])
        lenb = math.sqrt(self.metric_tensor[1,1])
        lenc = math.sqrt(self.metric_tensor[2,2])
        gamma = math.acos(self.metric_tensor[0,1]/lena/lenb)*360./(2.*math.pi)
        beta = math.acos(self.metric_tensor[0,2]/lena/lenc)*360./(2.*math.pi)
        alpha = math.acos(self.metric_tensor[1,2]/lenb/lenc)*360./(2.*math.pi)
        return lena, lenb, lenc, alpha, beta, gamma
   
    @property
    def kernel(self):
        try:
            return self._kernel
        except AttributeError:
            c = self.iter_cycles(node=self._graph.vertices()[0],
                                 used=[],
                                 nodes_visited=[],
                                 cycle_baggage=[],
                                 counter=0)
            zero_voltages = []
            for cycle in c:
                vect = np.zeros(self.shape)
                vect[self.return_indices(cycle)] = self.return_coeff(cycle)
                volt = self.get_voltage(vect)
                if np.allclose(np.abs(volt), np.zeros(3)):
                    zero_voltages.append(vect)
            self._kernel = np.concatenate((np.matrix(zero_voltages), self.cocycle), axis=0)
            return self._kernel

    @property
    def eon_projection(self):
        if self.kernel is None:
            return np.identity(self.shape)
        d = self.kernel*self.kernel.T
        sub_mat = np.matrix(self.kernel.T* d.I* self.kernel)
        return np.identity(self.shape) - sub_mat
   
    @property
    def projection(self):
        xx = (self.cycle_cocycle.I*self.periodic_rep)
        return xx*(xx.T*xx).I*xx.T
       
    @property
    def lattice_arcs(self):
        try:
            return self._lattice_arcs
        except AttributeError:
            self._lattice_arcs = self.get_arcs()
            return self._lattice_arcs

    @property
    def shape(self):
        return self._graph.size()

    @property
    def num_nodes(self):
        return self._graph.order() 

    @property
    def graph(self):
        return self._graph
    @graph.setter
    def graph(self, g):
        self._graph = DiGraph(g, multiedges=True, loops=True)

    @property
    def cycle_cocycle(self):
        try:
            return self._cycle_cocycle
        except AttributeError:
            self._cycle_cocycle = np.concatenate((self.cycle, self.cocycle),
                                                 axis = 0)
            return self._cycle_cocycle
