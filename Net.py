import math
import sys
from sage.all import *
import itertools
from uuid import uuid4
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize, anneal, brute, basinhopping, fsolve, root 
sys.path.append('/home/pboyd/lib/lmfit-0.7.2')
from lmfit import minimize, Parameters, Minimizer

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
                self.cocycle = self.add_to_matrix(vect, self.cocycle)

        if count != size:
            print "ERROR - could not find a linear independent cocycle basis!"
            sys.exit()
        # special case - pcu
        # NOTE : YOU WILL HAVE TO ADD 2 - coordinate nodes to pcu to get this to work!!!
        if size == 0:
            self.cocycle = None
            self.cocycle_rep = None
        else:
            self.cocycle = np.matrix(self.cocycle)
            self.cocycle_rep = np.matrix(np.zeros((size, self.ndim)))

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
        n = self.shape - self.num_nodes + 1
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
            if np.all(np.abs(volt) < 1.001) and check:
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
                             edge=None,
                             cycle=[],
                             used=[],
                             nodes_visited=[],
                             cycle_baggage=[],
                             counter=0)
        self.lattice_basis = np.zeros((self.ndim, self.shape))
        for cycle in c:
            vect = np.zeros(self.shape)
            try:
                vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            except IndexError:
                print vect
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
        if self.cocycle is not None:
            cocycle_rep = np.zeros((self.cocycle.shape[0], self.ndim))
            return self.cycle_cocycle.I*np.concatenate((self.cycle_rep, cocycle_rep),
                                                       axis=0)
        else:
            return self.cycle_cocycle.I*self.cycle_rep

    def min_function_scalar(self, cocyc_proj):
        cocyc_rep = np.reshape(cocyc_proj, (self.cocycle.shape[0], self.ndim))
        self.periodic_rep = np.concatenate((self.cycle_rep,cocyc_rep))
        M = self._lattice_arcs*\
                (self.lattice_basis*self.projection*self.lattice_basis.T)\
                *self._lattice_arcs.T

        nz = np.nonzero(self.colattice_dotmatrix)
        # SF?
        scale_factor = M.max()
        sol = np.sum(np.absolute(np.absolute(M[nz]/scale_factor) - np.absolute(self.colattice_dotmatrix[nz])))
        return sol
    
    def to_ind(self, str_obj):
        return tuple([int(i) for i in str_obj.split('_')[1:]])

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
        #M = self._lattice_arcs*\
        #        (self.lattice_basis*self.projection*self.lattice_basis.T)\
        #        *self._lattice_arcs.T
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
        sol = np.array(np.absolute(M[nz]) - np.absolute(self.colattice_dotmatrix[nz]))
        #sol = np.array(M[nz] - self.colattice_dotmatrix[nz])
        return sol.flatten()

    def min_function_array(self, cocyc_proj):
        def _obtain_dp_sum(self, v, M):
            edges = self.graph.outgoing_edges(v) + self.graph.incoming_edges(v)
            inds = self.return_indices(edges)
            combos = list(itertools.combinations_with_replacement(inds, 2))
            t = tuple(map(tuple, np.array(combos).T))
            #return np.sum(np.absolute(M[t]))
            return np.array(M[t]).flatten()[:self.ndim]

        cocyc_rep = np.reshape(cocyc_proj, (self.cocycle.shape[0], self.ndim))
        self.periodic_rep = np.concatenate((self.cycle_rep,cocyc_rep))
        M = self._lattice_arcs*\
                (self.lattice_basis*self.projection*self.lattice_basis.T)\
                *self._lattice_arcs.T

        # For some reason the root finding functions require that the 
        # return array be the same dimension as the input array...
        # WHY is a mystery to me.
        dp_sum = np.array([_obtain_dp_sum(i,M) for i in self.graph.vertices()[:-1]]).flatten()
        dd_ = np.array([_obtain_dp_sum(i,self.colattice_dotmatrix) for i in self.graph.vertices()[:-1]]).flatten()

        #nz = np.nonzero(self.colattice_dotmatrix)
        #sol = np.array(M[nz] - self.colattice_dotmatrix[nz]).flatten()
        sol = np.absolute(np.absolute(dp_sum) - np.absolute(dd_))
        return sol

    def init_params(self, init_guess):
        params = Parameters()
        for (i,j) in zip(*np.triu_indices_from(self.metric_tensor)):
            val = self.metric_tensor[i,j]
            if i == j:
                params.add("m_%i_%i"%(i,j), value=val, vary=False, min=0.001) # NOT SURE WHAT THE MIN/MAX should be here!
            else: 
                params.add("m_%i_%i"%(i,j), value=val, vary=False, min=-1, max=1) # NOT SURE WHAT THE MIN/MAX should be here!
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

    def get_embedding(self, init_guess=None):
        self.barycentric_embedding()
        if init_guess is None:
            init_guess = (np.zeros((self.num_nodes-1, self.ndim)))
        # questionable bounds there boyd.
        bounds = [(-1.,1.)]*(self.num_nodes-1)*self.ndim
        #bounds = None
        #cocycle = root(self.min_function_array,
        #               init_guess)
        # set up parameters class for the minimize function
        params = self.init_params(init_guess)
        #self.vary_cyc_coc_mt(params)
        #nz = np.nonzero(self.colattice_dotmatrix)
        #print self.colattice_dotmatrix[nz]
        #print (self.lattice_arcs*self.metric_tensor*self.lattice_arcs.T)[nz]/0.125 
        #for i in range(8):
        #    if i%2 == 0:
        #        self.vary_cocycle_rep(params)
        #    else:
        #        self.vary_metric_tensor(params)
        #    cocycle = minimize(self.min_function_lmfit, params, method='leastsq')
        #self.vary_metric_tensor(params)
        #self.vary_cocycle_rep(params)
        #self.vary_cyc_coc_mt(params)
        self.vary_coc_mt(params)
        #self.vary_all(params)
        #cocycle = minimize(self.min_function_lmfit, params, method='lbfgsb')
        cocycle = Minimizer(self.min_function_lmfit, params)
        cocycle.lbfgsb(factr=1.)
        #self.vary_coc_mt(params)
        #cocycle = minimize(self.min_function_lmfit, params, method='lbfgsb')
        #q = np.empty((self.cocycle.shape[0], self.ndim))
        q = np.empty((self.shape, self.ndim))
        mt = np.empty((self.ndim, self.ndim))
        for j in params:
            #print j, params[j].value
            if j[0] == 'm':
                i, k = self.to_ind(j)
                mt[i,k] = params[j].value
                mt[k,i] = params[j].value
            elif j[0] == 'c':
                q[self.to_ind(j)] = params[j].value

        #cocycle = minimize(self.min_function_scalar,
        #                   init_guess,
        #                   method="L-BFGS-B",
        #                   bounds=bounds,
        #                   options=self.lbfgsb_params
        #                   )
        #BRUTE DOES NOT SUPPORT VARIABLES > 32
        #cocycle = brute(self.min_function_scalar, bounds, Ns=20, finish=None, disp=True)

        #cocycle = basinhopping(self.min_function_scalar, init_guess)
        #q = cocycle.x
        #q = cocycle[0]
        return np.matrix(mt.copy()), np.matrix(q.copy())
        return np.concatenate((self.cycle_rep, 
                        np.reshape(q, (self.cocycle.shape[0],3))),
                        axis=0)

    def get_metric_tensor(self):
        self.metric_tensor = self.lattice_basis*self.projection*self.lattice_basis.T
        #self.metric_tensor = self.lattice_basis*self.eon_projection*self.lattice_basis.T

    def barycentric_embedding(self):
        if self.cocycle is not None:
            self.cocycle_rep = np.zeros((self.num_nodes-1, self.ndim))
            self.periodic_rep = np.concatenate((self.cycle_rep,
                                            self.cocycle_rep),
                                            axis = 0)
        else:
            self.periodic_rep = self.cycle_rep
        self.get_metric_tensor()
    
    def user_defined_embedding(self, cocycle_array):
        self.cocycle_rep = cocycle_array.reshape(self.num_nodes-1, self.ndim)
        self.periodic_rep = np.concatenate((self.cycle_rep, self.cocycle_rep))
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
        if len(pos.keys()) == self.graph.order() or not edges:
            # check if some of the nodes will naturally fall outside of the 
            # unit cell
            if len(pos.keys()) != self.graph.order():
                fgtn = set(self.graph.vertices()).difference(pos.keys())
                for node in fgtn:
                    poses = [e for e in bad_ones.keys() if node in e[:2]]
                    # TODO(pboyd): find an edge which already has been placed
                    # which corresponds to that node. then put it there
                    if poses:
                        # just take the first one.. who cares?
                        pos.update({node:bad_ones[poses[0]]})
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
                to_pos = coeff*np.array(self.lattice_arcs)[index] + pos[from_v]
                newedges = []
                if np.all(np.where((to_pos >= -0.00001) & (to_pos < 1.00001), True, False)):
                    pos.update({to_v:to_pos})
                    used.append(e)
                    ee = self.graph.outgoing_edges(to_v) + self.graph.incoming_edges(to_v)
                    newedges = [i for i in ee if i not in used and i not in edges]
                else:
                    bad_ones.update({e:to_pos})
                edges = newedges + edges[1:]
            else:
                used.append(e)
                edges = edges[1:]
            return self.vertex_positions(edges, used, pos, bad_ones)

    @property
    def kernel(self):
        try:
            return self._kernel
        except AttributeError:
            c = self.iter_cycles(node=self._graph.vertices()[0],
                                 edge=None,
                                 cycle=[],
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

    @property
    def anneal_params(self):
        return {
            'schedule'      : 'boltzmann',
            'maxfev'        : None,
            'maxiter'       : 500,
            'maxaccept'     : None,
            'ftol'          : 1e-6,
            'T0'            : None,
            'Tf'            : 1e-12,
            'boltzmann'     : 1.0,
            'learn_rate'    : 0.5,
            'quench'        : 1.0,
            'm'             : 1.0,
            'n'             : 1.0,
            'lower'         : -100,
            'upper'         : 100,
            'dwell'         : 250,
            'disp'          : True
            }

    @property
    def lbfgsb_params(self):
        return {
            'ftol'          : 0.00001,
            'gtol'          : 0.001,
            'maxiter'       : 15000,
            }
