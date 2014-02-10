from SecondaryBuildingUnit import SBU
from config import Terminate
from Net import Net
import sys
import itertools
from Visualizer import GraphPlot
from LinAlg import rotation_matrix, calc_angle, calc_axis
from Structure import Structure, Cell
import numpy as np
from logging import info, debug, warning, error
from copy import deepcopy
sys.path.append('/home/pboyd/lib/lmfit-0.7.2')
from lmfit import Minimizer, minimize, Parameters

np.set_printoptions(threshold=np.nan, precision=4, suppress=True, linewidth=185)

class Build(object):
    """Build a MOF from SBUs and a Net."""
    def __init__(self, options):
        self._net = None
        self.options = options
        self._sbus = []
        self.scale = 1.
        self._vertex_assign = {}
        self._edge_assign = {}
        self._sbu_degrees = None
        self._inner_product_matrix = None

    def _obtain_lattice_arcs(self):
        self._net.get_lattice_basis()
        self._net.get_cycle_basis()
        self._net.get_cocycle_basis()
        self._net.lattice_arcs

    def assign_vertices(self):
        """Assign SBUs to particular vertices in the graph"""
        for vert in self._net.graph.vertex_iterator():
            vert_deg = self._net.graph.degree(vert)
            sbu_match = [i for i in self._sbus if i.degree == vert_deg]
            # TODO(pboyd): assign sbus intelligently, based on edge lengths
            # and independent cycles in the net... ugh
            self._vertex_assign[vert] = deepcopy(sbu_match[0])

    def obtain_edge_vector(self, cp1, cp2):
        """Create an edge vector from two sbu 'connect points'"""
        e1 = self.vector_from_cp(cp1)
        e2 = self.vector_from_cp(cp2)
        len1 = np.linalg.norm(e1[:3])
        len2 = np.linalg.norm(e2[:3])
        dir = e1[:3]/len1
        e = dir*(len1+len2)
        return e

    def normalized_ipmatrix(self, vectors):
        norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
        v = vectors / norms.reshape(-1, 1)
        return np.inner(v,v)

    def assign_edge_labels(self, vertex):
        """Edge assignment is geometry dependent. This will try to 
        find the best assignment based on inner product comparison
        with the non-placed lattice arcs."""
        self._net.barycentric_embedding()
        local_arcs = self._vertex_assign[vertex].connect_points
        edges = self._net.graph.outgoing_edges(vertex) + \
                    self._net.graph.incoming_edges(vertex)
        indices = self._net.return_indices(edges)
        lattice_arcs = self._net.lattice_arcs[indices]
        #self._edge_assign[e] = (cp1.identifier, cp2.identifier)
        e_assign = {}
        vects = [self.vector_from_cp(cp) for cp in local_arcs]
        li = self.normalized_ipmatrix(vects)
        min=15000
        cc, assign = None, None
        #print "new batch"
        for e in itertools.permutations(edges):
            indices = self._net.return_indices(e)
            lattice_arcs = self._net.lattice_arcs[indices]*\
                    self._net.metric_tensor*self._net.lattice_arcs[indices].T
            max = lattice_arcs.max()
            la = np.empty((len(indices),len(indices)))
            for (i,j), val in np.ndenumerate(lattice_arcs):
                if i==j:
                    la[i,j] == val/max
                else:
                    v = val/np.sqrt(lattice_arcs[i,i])/np.sqrt(lattice_arcs[j,j])
                    la[i,j] = v
                    la[j,i] = v
            # don't know orientations yet!
            mm = np.sum(np.absolute(np.absolute(li)-np.absolute(la)))
            
            # using tensor product of the incidences
            coeff = np.array([-1. if j in self._net.graph.incoming_edges(vertex)
                               else 1. for j in e])
            td = np.tensordot(coeff, coeff, axes=0)
            mm = np.sum(np.absolute(np.multiply(li, td) - la))
            #mm = np.sum(np.absolute(li-la))
            #print mm
            if mm < min:
                cc = coeff
                min = mm
                assign = e
        #print "min = ", min
        # NB special MULT function for connect points
        return {e[2]:cp for (e,cp) in zip(assign, local_arcs)}

    def assign_edges(self):
        """Select edges from the graph to assign bonds between SBUs.
        This can become combinatorial...
        
        NB: if the SBUs have low symmetry, just selecting from a pool
        of SBU connection points may result in a node with the wrong
        orientation of edges.  There should be a better way of doing 
        this where the SBU geometry is respected.

        In this algorithm obtain the inner products of all the edges
        These will be used to later optimize the net to match the 
        SBUs.

        """
        done = False
        # create a pool of edges? to select from. This will gradually
        # disappear until all edges are assigned.
        # In cases where there are more than 3 edges, assignment can 
        # get tricky.
        edges = {}
        max_len = 0.

        for v in self._net.graph.vertex_iterator():
            for e, c in self.assign_edge_labels(v).items():
                self._edge_assign.setdefault(e, []).append((v,c))

        max_len = (None, 0.) 
        for e, ((v,cp1), (v,cp2)) in self._edge_assign.items():
            assert len(self._edge_assign[e]) == 2
            edge_vect = self.obtain_edge_vector(cp1, cp2)
            edges[e] = edge_vect
            leng = np.linalg.norm(edge_vect)
            if leng > max_len[1]:
                max_len = (int(e[1:])-1, leng)

        self.scale = max_len
        # obtain inner product matrix
        self._inner_product_matrix = np.zeros((self.net.shape, self.net.shape))
        for v in self._net.graph.vertex_iterator():
            loc_edges = self._net.graph.outgoing_edges(v) + \
                    self._net.graph.incoming_edges(v)
            # get edges oriented out from this node.
            sbu = self._vertex_assign[v]
            combos = itertools.combinations_with_replacement(loc_edges, 2)
            for (e1, e2) in combos:
                (i1, i2) = self._net.return_indices([e1, e2])
                cp_set1, cp_set2 = [self._edge_assign[e1[2]], self._edge_assign[e2[2]]]
                # correctly orient the edges to set the origin to the sbu of v
                # get edge1
                cp11 = cp_set1[0][1] if cp_set1[0][0] == v else cp_set1[1][1]
                cp12 = cp_set1[1][1] if cp_set1[1][0] != v else cp_set1[0][1]
                edge1 = self.obtain_edge_vector(cp11, cp12)
                # **********************MAY BREAK STUFF
                if e1 in self._net.graph.incoming_edges(v):
                    edge1 = -edge1
                # **********************MAY BREAK STUFF
                cp21 = cp_set2[0][1] if cp_set2[0][0] == v else cp_set2[1][1]
                cp22 = cp_set2[1][1] if cp_set2[1][0] != v else cp_set2[0][1]
                edge2 = self.obtain_edge_vector(cp21, cp22)
                # **********************MAY BREAK STUFF
                if e2 in self._net.graph.incoming_edges(v):
                    edge2 = -edge2
                # **********************MAY BREAK STUFF
                if i1 != i2:
                    en = np.dot(edge1, edge2)/np.linalg.norm(edge1)/np.linalg.norm(edge2)
                else:
                    en = np.dot(edge1, edge2)/max_len[1]/max_len[1]
                self._inner_product_matrix[i1,i2] = en 
                self._inner_product_matrix[i2,i1] = en
        self._inner_product_matrix = np.asmatrix(self._inner_product_matrix)
        
    def net_degrees(self):
        n = self._net.graph.degree_histogram()
        return sorted([i for i, j in enumerate(n) if j])

    def obtain_embedding(self):
        """Optimize the edges and cell parameters to obtain the crystal
        structure embedding.

        """
        # We first need to normalize the edge lengths of the net. This will be
        # done initially by setting the longest vector equal to the longest
        # vector of the barycentric embedding.
        self._net.colattice_dotmatrix = np.matrix(
                                        self._inner_product_matrix.copy())

        # this calls the optimization routine to match the tensor product matrix
        # of the SBUs and the net.
        self._net.metric_tensor, self._net.periodic_rep = self._net.get_embedding()
        la = self._net.cycle_cocycle.I*self._net.periodic_rep
        # obtain the length to normalize by from the custom embedding...
        msf = np.sqrt((la*self._net.metric_tensor*la.T)[self.scale[0], self.scale[0]])
        # self.scale stores the length of the longest SBU-SBU bond. The final
        # scale factor is len(SBU-SBU)/len(msf)
        sf = self.scale[1]/msf
        # the real metric tensor is obtained by multiplying by the scaling factor^2
        self._net.metric_tensor*=sf**2
        mt = self._net.metric_tensor
        self._net._lattice_arcs = la.copy()
        test = np.array([0.5, 0.5, 0.5])
        self.build_structure_from_net(test)
        self.show()
        #print self._net.metric_tensor

    def test_angle(self, index1, index2, mat):
        return np.arccos(mat[index1, index2]/np.sqrt(mat[index1, index1])/np.sqrt(mat[index2, index2]))*180./np.pi

    def custom_embedding(self, rep, mt):
        self._net.metric_tensor = np.matrix(mt)
        self._net.periodic_rep = np.matrix(rep)
        la = self._net.cycle_cocycle.I*rep
        ip = la*mt*la.T
        ipsbu = self._inner_product_matrix
        nz = np.nonzero(np.triu(ipsbu))

        self.build_structure_from_net(np.zeros(self._net.ndim))
        self.show()

    def build_structure_from_net(self, init_placement):
        """Orient SBUs to the nodes on the net, create bonds where needed, etc.."""
        # get the real cell
        # get the real edges
        struct = Structure(self.options, name="test", params=self._net.get_3d_params())
        cell = struct.cell.lattice
        V = self.net.graph.vertices()[0] 
        edges = self.net.graph.outgoing_edges(V) + self.net.graph.incoming_edges(V)
        sbu_pos = self._net.vertex_positions(edges, [], pos={V:init_placement})
        for v in self._net.graph.vertex_iterator():
            self.sbu_orient(v, cell)
            fc = sbu_pos[v]
            tv = np.dot(fc, cell)
            self.sbu_translate(v, tv)
            struct.add_sbu(self._vertex_assign[v])

        struct.write_cif()

    def rotation_function(self, params, sbu_vects, data):
        axis = np.array((params['a1'].value, params['a2'].value, params['a3'].value))
        angle = params['angle'].value
        R = rotation_matrix(axis, angle)
        #res = np.dot(sbu_vects, R[:3,:3])
        res = np.dot(R[:3,:3], sbu_vects.T)
        norms = np.apply_along_axis(np.linalg.norm, 1, res.T)
        v = res.T / norms.reshape(-1, 1)
        angles = np.array([calc_angle(v1, v2) for v1, v2 in zip(v, data)])
        print angles
        return (v - data).flatten()
        #return angles 

    def sbu_orient(self, v, cell):
        """Optimize the rotation to match vectors"""
        sbu = self._vertex_assign[v]
        edges = self._net.graph.outgoing_edges(v) + self._net.graph.incoming_edges(v)
        # re index the edges to match the order of the connect points in the sbu list
        #print "SBU", sbu.name
        indexed_edges = []
        for cp in sbu.connect_points:
            for e in edges:
                if e in self._net.graph.outgoing_edges(v):
                    coeff = 1.
                else:
                    coeff = -1.
                (v1, cp1),(v2, cp2) = self._edge_assign[e[2]]
                if v1 == v and cp1.identifier == cp.identifier:
                    indexed_edges.append((coeff, e))
                elif v2 == v and cp2.identifier == cp.identifier:
                    indexed_edges.append((coeff, e))
        if len(indexed_edges) != sbu.degree:
            error("There was an error assigning edges "+
                        "to the sbu %s"%(sbu.name))
            Terminate(errcode=1)
        inds = self._net.return_indices([m[1] for m in indexed_edges])
        coefficients = np.array([m[0] for m in indexed_edges])
        arcs = np.dot(self._net.lattice_arcs[inds], cell)
        norms = np.apply_along_axis(np.linalg.norm, 1, arcs)
        # get the right orientation of the arcs (all pointing away from the node)
        # **********************MAY BREAK STUFF
        arcs = np.array(arcs / norms.reshape(-1, 1)) * coefficients[:,None]
        # **********************MAY BREAK STUFF 
        print "Normalized inner product matrix of the arcs from the net embedding"
        print np.matrix(arcs)*np.matrix(arcs).T
        sbu_vects = np.array([self.vector_from_cp(cp) 
                                for cp in sbu.connect_points])
        norms = np.apply_along_axis(np.linalg.norm, 1, sbu_vects)
        sbu_vects = sbu_vects / norms.reshape(-1, 1)
        print "Normalized inner product matrix of the sbu vectors"
        print np.matrix(sbu_vects)*np.matrix(sbu_vects).T
        params = Parameters()
        params.add('a1', value=0.001, min=-1., max=1.)
        params.add('a2', value=0.001, min=-1., max=1.)
        params.add('a3', value=0.001, min=-1., max=1.)
        params.add('angle', value=0.1, min=0., max=np.pi)
        min = Minimizer(self.rotation_function, params, fcn_args=(sbu_vects, arcs))
        #min.lbfgsb(factr=10., epsilon=1e-5, pgtol=1e-4)
        min.leastsq(xtol=1.e-5, ftol=1.e-6)
        #minimize(self.rotation_function, params, args=(sbu_vects, arcs), method='lbfgsb')
        axis = np.array([params['a1'].value, params['a2'].value, params['a3'].value])
        angle = params['angle'].value
        R = rotation_matrix(axis, angle)
        sbu.rotate(R)

    def sbu_translate(self, v, trans):
        sbu = self._vertex_assign[v]
        sbu.translate(trans)

    def show(self):
        g = GraphPlot(self._net)
        #g.view_graph()
        g.view_placement(init=(0.2, 0.2, 0.3))
    
    def vector_from_cp(self,cp):
        return cp.origin[:3].copy()# + cp.z[:3]

    @property
    def check_net(self):
        if self._net.shape < 25 and self.sbu_degrees == self.net_degrees():
            return True
        return False

    @property
    def sbu_degrees(self):
        if self._sbu_degrees is not None:
            return self._sbu_degrees
        else:
            deg = [i.degree for i in self._sbus]
            lin = [i.linear for i in self._sbus]
            self._sbu_degrees = sorted([j for i,j in zip(lin,deg) if not i])
            return self._sbu_degrees

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, (graph, volt)):
        self._net = Net(graph)
        self._net.voltage = volt
        self._obtain_lattice_arcs()

    @property
    def sbus(self):
        return self._sbus

    @sbus.setter
    def sbus(self, sbus):
        self._sbus = sbus

