import math
import sys
from sage.all import *
import itertools
from uuid import uuid4
from logging import info, debug, warning, error
import numpy as np
from LinAlg import DEG2RAD
#from scipy.optimize import fmin_l_bfgs_b, minimize, anneal, brute, basinhopping, fsolve, root 
sys.path.append('/home/pboyd/lib/lmfit-0.7.2')
from lmfit import minimize, Parameters, Minimizer, report_errors
from config import Terminate

class SystreDB(dict):
    """A dictionary which reads a file of the same format read by Systre"""
    def __init__(self, filename=None):
        self.voltages = {}
        self.read_store_file(filename)
        # scale holds the index and value of the maximum length^2 for the 
        # real vectors associated with edges of the net.  This is only
        # found after SBUs have been assigned to nodes and edges.
        self.scale = (None, None)

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
        self.name = None
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
            # Keep an original for reference purposes.
            self.original_graph = DiGraph(graph, multiedges=True, loops=True)

    def get_cocycle_basis(self):
        """The orientation is important here!"""
        size = self._graph.order() - 1
        length = self._graph.size()
        count = 0
        for vert in self._graph.vertices():
            if count == size:
                break
            vect = np.zeros(length)
            out_edges = self._graph.outgoing_edges(vert)
            inds = self.return_indices(out_edges)
            if inds:
                vect[inds] = 1.
            in_edges = self._graph.incoming_edges(vert)
            inds = self.return_indices(in_edges)
            if inds:
                vect[inds] = -1.
            if self.cycle_cocycle_check(vect):# or len(self.neighbours(vert)) == 2:
                count += 1
                self.cocycle = self.add_to_matrix(vect, self.cocycle)

        if count != size:
            print "ERROR - could not find a linearly independent cocycle basis!"
            Terminate(errcode=1) 
        # special case - pcu
        # NOTE : YOU WILL HAVE TO ADD 2 - coordinate nodes to pcu to get this to work!!!
        if size == 0:
            self.cocycle = None
            self.cocycle_rep = None
        else:
            self.cocycle = np.matrix(self.cocycle)
            self.cocycle_rep = np.matrix(np.zeros((size, self.ndim)))

    def add_name(self):
        try:
            name = chr(self.order + ord("A"))
        except ValueError:
            name = str(self.order)
        return name

    def insert_and_join(self, vfrom, vto, edge_label=None):
        if edge_label is None:
            edge_label = "e%i"%(self.shape)
        self.graph.add_vertex(vto)
        edge = (vfrom, vto, edge_label)
        self.graph.add_edge(vfrom, vto, edge_label)
        return edge

    def add_edges_between(self, edge, N):
        newedges = []
        V1 = edge[0] if edge in self.graph.outgoing_edges(edge[0]) \
                else edge[1]
        V2 = edge[1] if edge in self.graph.incoming_edges(edge[1]) \
                else edge[0]
        name = self.add_name()
        newedges.append(self.insert_and_join(V1, name, edge_label=edge[2]))
        vfrom = name
        d = self.ndim
        newnodes = []
        for i in range(N-1):
            newnodes.append(vfrom)
            name = self.add_name()
            newedges.append(self.insert_and_join(vfrom, name))
            vfrom = name
            self.voltage = np.concatenate((self.voltage,np.zeros(d).reshape(1,d)))
        # final edge to V2
        newnodes.append(V2)
        lastedge = (vfrom, V2, "e%i"%(self.shape))
        newedges.append(lastedge)
        self.graph.add_edge(vfrom, V2, "e%i"%(self.shape))
        self.graph.delete_edge(edge)
        self.voltage = np.concatenate((self.voltage,np.zeros(d).reshape(1,d)))
        return newnodes, newedges

    def cycle_cocycle_check(self, vect):
        if self.cocycle is None and self.cycle is None:
            return True
        elif self.cocycle is None and self.cycle is not None:
            return self.check_linear_dependency(vect, self.cycle)
        else:
            return self.check_linear_dependency(vect, 
                                self.add_to_matrix(self.cocycle, self.cycle))

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
                             edge=None,
                             cycle=[],
                             used=[],
                             nodes_visited=[],
                             cycle_baggage=[],
                             counter=0)
        n = self.shape - self.order + 1
        count = 0
        if self.lattice_basis is not None:
            self.cycle = self.add_to_matrix(self.lattice_basis, self.cycle)
            self.cycle_rep = self.add_to_matrix(np.identity(self.ndim), self.cycle_rep)
            count += self.ndim
        for id, cycle in enumerate(c):
            if count >= n:
                break
            vect = np.zeros(self.shape)
            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            volt = self.get_voltage(vect)
            # REPLACE WITH CHECK_LINEAR_DEPENDENCY()
            check = self.cycle_cocycle_check(vect)
            if np.all(np.abs(volt) < 1.001) and np.sum(np.abs(volt) > 0.) and check:
                self.cycle = self.add_to_matrix(vect, self.cycle)
                self.cycle_rep = self.add_to_matrix(volt, self.cycle_rep)
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

    def iter_tree(self, vertex, edge, edges, depth=False):
        new = [e for e in edges if e[0] == vertex or e[1] == vertex and e != edge]
        for edge in new:
            iter_trie

    def simple_cycle_basis(self):
        """Cycle basis is constructed using a minimum spanning tree.
        This tree is traversed, and all the remaining edges are added
        to obtain the basis.

        """
        edges = self.graph.edges()
        mspt = self.graph.to_undirected().min_spanning_tree()
        tree = Graph(mspt, multiedges=False, loops=False)
        #self.graph.show()
        cycle_completes = [i for i in edges if i not in mspt and (i[1], i[0], i[2]) not in mspt]
        self.cycle = []
        self.cycle_rep = []
        for (v1, v2, e) in cycle_completes:
            path = tree.shortest_path(v1, v2)
            basis_vector = np.zeros(self.shape)
            cycle, coefficients = [], []
            for pv1, pv2 in itertools.izip(path[:-1], path[1:]):
                edge = [i for i in tree.edges_incident([pv1, pv2]) if pv1 in i[:2] and pv2 in i[:2]][0]
                if edge not in edges:
                    edge = (edge[1], edge[0], edge[2])
                    if edge not in edges:
                        error("Encountered an edge (%s, %s, %s) not in "%(edge) +
                        " the graph while finding the basis of the cycle space!")
                        Terminate(errcode=1)
                coeff = 1. if edge in self.graph.outgoing_edges(pv1) else -1.
                coefficients.append(coeff)
                cycle.append(edge)
            # opposite because we are closing the loop. i.e. going from v2 back to v1
            edge = (v1, v2, e) if (v1, v2, e) in edges else (v2, v1, e)
            coeff = 1. if edge in self.graph.incoming_edges(v1) else -1.
            coefficients.append(coeff)
            cycle.append(edge)
            basis_vector[self.return_indices(cycle)] = coefficients
            voltage = self.get_voltage(basis_vector)
            self.cycle.append(basis_vector)
            self.cycle_rep.append(voltage)
            #cycle = [xx for xx in tree.edges_incident(path) if xx[0] in path and xx[1] in path] + [(v1, v2, e)]
        #print len(cycle_completes)
        #print self.graph.size() - (self.graph.order() - 1)
        self.cycle = np.matrix(np.array(self.cycle))
        self.cycle_rep = np.matrix(np.array(self.cycle_rep))
        #self.graph.show(edge_labels=True)
        #self.graph.show3d(iterations=500)
        #raw_input("Press [Enter]\n")

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
            #yield c
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

    def get_lattice_basis(self):
        L = []   
        for i in np.array(self.cycle_rep):
            L.append(vector(QQ, i.tolist()))
        V = QQ**self.ndim
        lattice = []
        for e in np.identity(self.ndim):
            ev = vector(e)
            L.append(ev)
            
            #vect = (V.linear_dependence(L, zeros='left')[-1][:-1])
            #nz = np.nonzero(vect)
            mincount = self.shape
            vect = None
            for jj in V.linear_dependence(L, zeros='left'):
                if not np.allclose(jj[-1], 0):
                    vv = jj[:-1]*-1.*jj[-1]
                    nz = np.nonzero(vv)
                    tv = np.sum(np.array(self.cycle)[nz] * np.array(vv)[nz][:, None], axis=0)
                    if len(nz) == 1:
                        vect = tv 
                        break
                    elif len(nz) < mincount and self.is_integral(tv):
                        vect = tv 
                        mincount = len(nz)
            lattice.append(tv)
            L.pop(-1)
        self.lattice_basis = np.matrix(np.array(lattice))
        #print self.lattice_basis

    #def get_lattice_basis(self):
    #    """Obtains a lattice basis by iterating over all the cycles and finding
    #    ones with net voltages satisifying one of the n dimensional basis vectors.

    #    """
    #    basis_vectors = []
    #    c = self.iter_cycles(node=self._graph.vertices()[0],
    #                         edge=None,
    #                         cycle=[],
    #                         used=[],
    #                         nodes_visited=[],
    #                         cycle_baggage=[],
    #                         counter=0)
    #    self.lattice_basis = np.zeros((self.ndim, self.shape))
    #    for cycle in c:
    #        vect = np.zeros(self.shape)
    #        try:
    #            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
    #        except IndexError:
    #            print vect
    #        volt = self.get_voltage(vect)
    #        for id, e in enumerate(np.identity(self.ndim)):
    #            if np.allclose(np.abs(volt), e):
    #                check = self.check_linear_dependency(vect, self.lattice_basis[basis_vectors])
    #                if id not in basis_vectors and check:
    #                    basis_vectors.append(id)
    #                    self.lattice_basis[id] = volt[id]*vect
    #                elif (np.count_nonzero(vect) < np.count_nonzero(self.lattice_basis[id])) and check:
    #                    self.lattice_basis[id] = volt[id]*vect
    #    if len(basis_vectors) != self.ndim:
    #        print "ERROR: could not find all cycle vectors for the lattice basis!"
    #        Terminate(errcode=1)

    def check_linear_dependency(self, vect, vset):
        if not np.any(vset):
            return True
        else:
            A = np.concatenate((vset, np.reshape(vect, (1, self.shape))))
        lrank = vset.shape[0] + 1
        #U, s, V = np.linalg.svd(A)
        #if np.all(s > 0.0001):
        if np.linalg.matrix_rank(A) == lrank:
            return True
        return False

    def get_index(self, edge):
        return int(edge[2][1:])-1

    def return_indices(self, edges):
        return [self.get_index(i) for i in edges]

    def return_coeff(self, edges):
        assert edges[0][3]
        return [i[3] for i in edges]

    def to_ind(self, str_obj):
        return tuple([int(i) for i in str_obj.split('_')[1:]])

    def init_params(self, init_guess):
        params = Parameters()
        self.barycentric_embedding()
        for (i,j) in zip(*np.triu_indices_from(self.metric_tensor)):
            val = self.metric_tensor[i,j]
            #val = np.identity(3)[i,j]
            if i == j:
                params.add("m_%i_%i"%(i,j), value=val, vary=False, min=0.001) # NOT SURE WHAT THE MIN/MAX should be here!
            else:
                #NB the inner products of <a,b> should not be limited to the range [-1,1]! The code is well behaved
                # with these constraints so un-comment the line below if things go awry.
                #params.add("m_%i_%i"%(i,j), value=val, vary=False, min=-1, max=1) # NOT SURE WHAT THE MIN/MAX should be here!
                params.add("m_%i_%i"%(i,j), value=val, vary=False)
        for (i,j), val in np.ndenumerate(self.cycle_rep.copy()):
            if val == 0.:
                params.add("cy_%i_%i"%(i,j), value=val, vary=False, min=0, max=1)
            elif val == 1.:
                params.add("cy_%i_%i"%(i,j), value=val, vary=False, min=0, max=1)
            elif val == -1.:
                params.add("cy_%i_%i"%(i,j), value=val, vary=False, min=-1, max=0)
        pp = self.cycle_rep.shape[0]
        for (i,j), val in np.ndenumerate(init_guess):
            params.add("co_%i_%i"%(i+pp,j), value=val, vary=False, min=-1, max=1)
        return params

    def vary_cycle_rep_intonly(self, params):
        for p in params:
            if p.split("_")[0] == 'cy':
                i,j = self.to_ind(p)
                if abs(self.cycle_rep[i,j]) == 1.:
                    params[p].vary = True
                else:
                    params[p].vary = False
            else:
                params[p].vary = False
    
    def vary_cycle_rep_all(self, params):
        for p in params:
            if p.split("_")[0] == 'cy':
                params[p].vary = True
            else:
                params[p].vary = False

    def vary_metric_tensor(self, params):
        for p in params:
            if p[0] == 'm':
                params[p].vary = True
            else:
                params[p].vary = False

    def vary_cocycle_rep(self, params):
        for p in params:
            if p.split("_")[0] == 'co':
                params[p].vary = True
            else:
                params[p].vary = False

    def vary_coc_mt(self, params):
        for p in params:
            if p.split("_")[0] == "co":
                params[p].vary = True
            elif p[0] == 'm':
                params[p].vary = True
            else:
                params[p].vary = False

    def vary_cyc_coc_mt(self, params):
        for p in params:
            if p.split("_")[0] == "cy":
                i,j = self.to_ind(p)
                if abs(self.cycle_rep[i,j]) == 1.:
                    params[p].vary = True
                else:
                    params[p].vary = False
            else:
                params[p].vary = True
    
    def vary_all(self, params):
        for p in params:
            params[p].vary = True
    
    def min_function_lmfit(self, params):
        rep = np.matrix(np.zeros((self.shape, self.ndim)))
        mt = np.matrix(np.zeros((self.ndim,self.ndim)))
        for p in params:
            if p[0] == 'm':
                i,j = self.to_ind(p)
                mt[i,j] = params[p].value
                mt[j,i] = params[p].value
            elif p[0] == 'c':
                rep[self.to_ind(p)] = params[p].value
        la = self.cycle_cocycle.I*rep
        M = la*mt*la.T
        scale_factor = M.max()
        for (i, j) in zip(*np.triu_indices_from(M)):
            val = M[i,j]
            if i != j:
                v = val/np.sqrt(M[i,i])/np.sqrt(M[j,j])
                M[i,j] = v
                M[j,i] = v
        for i, val in np.ndenumerate(np.diag(M)):
            M[i,i] = val/scale_factor
        nz = np.nonzero(np.triu(self.colattice_dotmatrix))
        sol = (np.array(M[nz] - self.colattice_dotmatrix[nz]))
        return sol.flatten()

    def assign_ip_matrix(self, mat):
        """Get the colattice dot matrix from Builder.py. This is an inner 
        product matrix of all the SBUs assigned to particular nodes.
        """
        max_ind, max_val = np.diag(mat).argmax(), np.diag(mat).max()
        self.scale = (max_ind, max_val)
        # this sbu_tensor_matrix is probably not needed...
        self.sbu_tensor_matrix = mat
        self.colattice_dotmatrix = np.zeros((mat.shape[0], mat.shape[1]))
        for (i, j) in zip(*np.triu_indices_from(mat)):
            if i == j:
                self.colattice_dotmatrix[i,j] = mat[i,j]/max_val
            else:
                val = mat[i,j] / np.sqrt(mat[i,i]) / np.sqrt(mat[j,j])
                self.colattice_dotmatrix[i,j] = val 
                self.colattice_dotmatrix[j,i] = val

    def get_embedding(self, init_guess=None):
        if init_guess is None:
            init_guess = (np.zeros((self.order-1, self.ndim)))
        # set up parameters class for the minimize function
        params = self.init_params(init_guess)
        self.vary_coc_mt(params)
        #self.vary_cocycle_rep(params)
        #minimize(self.min_function_lmfit, params, method='Newton-CG')
        min = Minimizer(self.min_function_lmfit, params)
        min.lbfgsb(factr=1000., epsilon=1e-6, pgtol=1e-6)
        #min.leastsq(xtol=1.e-7, ftol=1.e-7)
        fit = self.min_function_lmfit(params)
        self.report_errors(fit)
        #print report_errors(params)
        q = np.empty((self.shape, self.ndim))
        mt = np.empty((self.ndim, self.ndim))
        for j in params:
            if j[0] == 'm':
                i, k = self.to_ind(j)
                mt[i,k] = params[j].value
                mt[k,i] = params[j].value
            elif j[0] == 'c':
                q[self.to_ind(j)] = params[j].value
        
        self.periodic_rep = q
        self.metric_tensor = mt
        la = self.lattice_arcs
        scind = self.scale[0]
        sclen = self.scale[1]
        self.scale_factor = sclen/np.diag(self.lattice_arcs*self.metric_tensor*self.lattice_arcs.T)[scind]
        self.metric_tensor *= self.scale_factor
        #print la*mt*la.T
        #print self.sbu_tensor_matrix

    def report_errors(self, fit):
        edge_lengths = []
        angles = []
        nz = np.nonzero(np.triu(np.array(self.colattice_dotmatrix)))
        count = 0
        for (i, j) in zip(*nz):
            if i != j:
                angles.append(fit[count])
            else:
                edge_lengths.append(fit[count])
            count += 1
        edge_average, edge_std = np.mean(edge_lengths), np.std(edge_lengths)
        debug("Average error in edge length: %12.5f +/- %9.5f Angstroms"%(
                                    math.copysign(1, edge_average)*
                                    np.sqrt(abs(edge_average)*self.scale[1]),
                                    math.copysign(1, edge_std)*
                                    np.sqrt(abs(edge_std)*self.scale[1])))
        angle_average, angle_std = np.mean(angles), np.std(angles)
        debug("Average error in edge angles: %12.5f +/- %9.5f degrees"%(
                        angle_average/DEG2RAD, angle_std/DEG2RAD))

    def get_metric_tensor(self):
        #self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        self.metric_tensor = self.lattice_basis*self.eon_projection*self.lattice_basis.T

    def barycentric_embedding(self):
        if self.cocycle is not None:
            self.cocycle_rep = np.zeros((self.order-1, self.ndim))
            self.periodic_rep = np.concatenate((self.cycle_rep,
                                            self.cocycle_rep),
                                            axis = 0)
        else:
            self.periodic_rep = self.cycle_rep
        self.get_metric_tensor()
    
    def get_2d_params(self):
        self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        lena=math.sqrt(self.metric_tensor[0,0])
        lenb=math.sqrt(self.metric_tensor[1,1])
        gamma=math.acos(self.metric_tensor[1,0]/lena/lenb)
        return lena, lenb, gamma

    def get_3d_params(self):
        lena = math.sqrt(self.metric_tensor[0,0])
        lenb = math.sqrt(self.metric_tensor[1,1])
        lenc = math.sqrt(self.metric_tensor[2,2])
        gamma = math.acos(self.metric_tensor[0,1]/lena/lenb)
        beta = math.acos(self.metric_tensor[0,2]/lena/lenc)
        alpha = math.acos(self.metric_tensor[1,2]/lenb/lenc)
        return lena, lenb, lenc, alpha, beta, gamma
    
    def vertex_positions(self, edges, used, pos={}, bad_ones = {}):
        """Recursive function to find the nodes in the unit cell.
        How it should be done:

        Create a growing tree around the init placed vertex. Evaluate
        which vertices wind up in the unit cell and place them.  Continue
        growing from those vertices in the unit cell until all are found.
        """
        # NOTE: NOT WORKING - FIX!!!
        lattice_arcs = self.lattice_arcs
        if len(pos.keys()) == self.graph.order():
            return pos
        else:
            # generate all positions from all edges growing outside of the current vertex
            # iterate through each until an edge is found which leads to a vertex in the 
            # unit cell.
            e = edges[0]
            if e[0] not in pos.keys() and e[1] not in pos.keys():
                pass
            elif e[0] not in pos.keys() or e[1] not in pos.keys():
                from_v = e[0] if e[0] in pos.keys() else e[1]
                to_v = e[1] if e[1] not in pos.keys() else e[0]
                coeff = 1. if e in self.graph.outgoing_edges(from_v) else -1.
                index = self.get_index(e)
                to_pos = coeff*np.array(lattice_arcs)[index] + pos[from_v]
                newedges = []
                #FROM HERE REMOVED IN-CELL CHECK
                to_pos = np.array([i%1 for i in to_pos])
                pos.update({to_v:to_pos})
                used.append(e)
                ee = self.graph.outgoing_edges(to_v) + self.graph.incoming_edges(to_v)
                newedges = [i for i in ee if i not in used and i not in edges]
                edges = newedges + edges[1:]
            else:
                used.append(e)
                edges = edges[1:]
            return self.vertex_positions(edges, used, pos, bad_ones)

    def indices_with_voltage(self, volt):
        return np.where([np.all(i==volt) for i in np.array(self.cycle_rep)])
    
    def is_integral(self, vect):
        return np.all(np.equal(np.mod(vect,1),0))
        #return np.all(np.logical_or(np.abs(vect) == 0., np.abs(vect) == 1.))

    @property
    def kernel(self):
        if hasattr(self, '_kernel'):
            return self._kernel
        kernel_vectors = []
        max_count = self.shape - self.ndim - (self.order-1)
        # if no kernel vectors need to be found, just return
        # the cocycle vectors.
        if max_count == 0:
            self._kernel = self.cocycle.copy()
            return self._kernel    
        self._kernel = None
        # obtain a basis of the cycle voltages
        L = []
        for i in np.array(self.cycle_rep):
            L.append(vector(QQ, i.tolist()))
        V = QQ**self.ndim
        for v in V.linear_dependence(L, zeros='left'):
            nz = np.nonzero(v)
            # obtain the linear combination of cycle vectors
            cv_comb =  np.array(self.cycle)[nz] * np.array(v)[nz][:, None]
            if self.is_integral(np.sum(cv_comb, axis=0)):
                kernel_vectors.append(np.sum(cv_comb, axis=0))
            if len(kernel_vectors) >= max_count:
                break
        # if not enough kernel vectors were found from the cycle basis, 
        # then iterate over cycles which are linearly independent from
        # the vectors already in the kernel. NB: this is fucking slow.
        if len(kernel_vectors) != max_count:
            warning("The number of vectors in the kernel does not match the size of the graph!")
            c = self.iter_cycles(node=self._graph.vertices()[0],
                                 edge=None,
                                 cycle=[],
                                 used=[],
                                 nodes_visited=[],
                                 cycle_baggage=[],
                                 counter=0)
            while len(kernel_vectors) < max_count:
                try:
                    cycle = c.next()
                except StopIteration:
                    # give up, use the cocycle basis
                    self._kernel = self.cocycle.copy()
                    return self._kernel
                vect = np.zeros(self.shape)
                vect[self.return_indices(cycle)] = self.return_coeff(cycle)
                volt = self.get_voltage(vect)
                if np.allclose(np.abs(volt), np.zeros(3)) and\
                        self.check_linear_dependency(vect, np.array(kernel_vectors)):
                    kernel_vectors.append(vect)

        self._kernel = np.concatenate((np.matrix(kernel_vectors), self.cocycle))
        return self._kernel
    #@property
    #def kernel(self):
    #    """In the case of a 3-periodic net, one may choose 3 cycles of 
    #    the quotient with independent net voltages. The net voltage of
    #    the remaining cycles in a cycle basis of the quotient graph 
    #    may then be written as a combination of these voltages, thus
    #    providing as many cycle vectors with zero net voltage.

    #    """
    #    try:
    #        return self._kernel
    #    except AttributeError:
    #        c = self.iter_cycles(node=self._graph.vertices()[0],
    #                             edge=None,
    #                             cycle=[],
    #                             used=[],
    #                             nodes_visited=[],
    #                             cycle_baggage=[],
    #                             counter=0)
    #        zero_voltages = []
    #        max_count = self.shape - self.ndim - self.cocycle.shape[0]
    #        count = 0
    #        for cycle in c:
    #            if count == max_count:
    #                break
    #            vect = np.zeros(self.shape)
    #            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
    #            volt = self.get_voltage(vect)
    #            if np.allclose(np.abs(volt), np.zeros(3)) and \
    #                    self.check_linear_dependency(vect, np.array(zero_voltages)):
    #                zero_voltages.append(vect)
    #                count += 1
    #        if not zero_voltages:
    #            self._kernel = self.cocycle.copy()
    #        else:
    #            self._kernel = np.concatenate((np.matrix(zero_voltages), self.cocycle), axis=0)
    #        return self._kernel

    def neighbours(self, vertex):
        return self.graph.outgoing_edges(vertex) + self.graph.incoming_edges(vertex)

    @property
    def eon_projection(self):
        if self.cocycle is not None:
            d = self.kernel*self.kernel.T
            sub_mat = np.matrix(self.kernel.T* d.I* self.kernel)
            return np.identity(self.shape) - sub_mat
        # if the projection gets here this is a minimal embedding
        return np.identity(self.shape)

    @property
    def projection(self):
        la = self.lattice_arcs
        return la*(la.T*la).I*la.T
       
    @property
    def lattice_arcs(self):
        return self.cycle_cocycle.I*self.periodic_rep

    @property
    def shape(self):
        return self._graph.size()

    @property
    def order(self):
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
            if self.cocycle is None and self.cycle is None:
                raise AttributeError("Both the cycle and cocycle "+
                                    "basis have not been allocated")
            elif self.cocycle is None:
                self._cycle_cocycle = self.cycle.copy()

            elif self.cycle is None:
                raise AttributeError("The cycle "+
                                    "basis has not been allocated")
            else:
                self._cycle_cocycle = np.concatenate((self.cycle, self.cocycle))
            return self._cycle_cocycle

