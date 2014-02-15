from SecondaryBuildingUnit import SBU
from config import Terminate
from Net import Net
import sys
import itertools
from Visualizer import GraphPlot
from LinAlg import rotation_from_vectors, rotation_matrix, rotation_from_omega, calc_angle, calc_axis, DEG2RAD
from LinAlg import central_moment, raw_moment, get_CI, elipsoid_vol 
from Structure import Structure, Cell
import numpy as np
from logging import info, debug, warning, error
from copy import deepcopy
sys.path.append('/home/pete/lib/lmfit-0.7.2')
from lmfit import Minimizer, minimize, Parameters, report_errors

np.set_printoptions(threshold=np.nan, precision=4, suppress=True, linewidth=185)

class Build(object):
    """Build a MOF from SBUs and a Net."""
    def __init__(self, options):
        self._net = None
        self.options = options
        self._sbus = []
        self.scale = 1.
        self._vertex_sbu = {}
        self._edge_assign = {}
        self._sbu_degrees = None
        self._inner_product_matrix = None

    def _obtain_cycle_bases(self):
        self._net.get_lattice_basis()
        self._net.get_cycle_basis()
        self._net.get_cocycle_basis()

    def assign_vertices(self):
        """Assign SBUs to particular vertices in the graph"""
        # TODO(pboyd): assign sbus intelligently, based on edge lengths
        # and independent cycles in the net... ugh
        for vert in self.sbu_vertices:
            # is there a way to determine the symmetry operations applicable
            # to a vertex?
            # if so, we could compare with SBUs...
            vert_deg = self._net.graph.degree(vert)
            sbu_match = [i for i in self._sbus if i.degree == vert_deg]
            # match tensor product matrices
            if len(sbu_match) > 1:
                self._vertex_sbu[vert] = self.select_sbu(vert, sbu_match)
            else:
                self._vertex_sbu[vert] = deepcopy(sbu_match[0])
            self._vertex_sbu[vert].vertex_id = vert

    def select_sbu(self, v, sbus):
        """This is a hackneyed way of selecting the right SBU,
        will use until it breaks something.

        """
        edges = self._net.graph.outgoing_edges(v) + self._net.graph.incoming_edges(v)
        indices = self._net.return_indices(edges)
        lattice_arcs = self._net.lattice_arcs[indices]
        ipv = lattice_arcs*self._net.metric_tensor*lattice_arcs.T
        ipv = self.scaled_ipmatrix(ipv)
        # just take the max and min angles... 
        inds = np.triu_indices(ipv.shape[0], k=1) 
        max, min = np.absolute(ipv[inds]).max(), np.absolute(ipv[inds]).min()
        minmag = 15000.
        #FIXME(pboyd): no consideration for multiple SBUs.
        for sbu in sbus:
            vects = np.array([self.vector_from_cp_SBU(cp, sbu) for cp in 
                              sbu.connect_points])
            ipc = self.scaled_ipmatrix(np.inner(vects, vects))
            imax, imin = np.absolute(ipc[inds]).max(), np.absolute(ipc[inds]).min()
            mm = np.sum(np.absolute([max-imax, min-imin]))
            if mm < minmag:
                minmag = mm
                assign = sbu
        return deepcopy(assign)

    def obtain_edge_vector(self, cp1, cp2):
        """Create an edge vector from two sbu 'connect points'"""
        e1 = self.vector_from_cp(cp1)
        e2 = self.vector_from_cp(cp2)
        len1 = np.linalg.norm(e1[:3])
        len2 = np.linalg.norm(e2[:3])
        dir = e1[:3]/len1
        e = dir*(len1+len2)
        return e

    def scaled_ipmatrix(self, ipmat):
        """Like normalized inner product matrix, however the 
        diagonal is scaled to the longest vector."""
        ret = np.empty_like(ipmat)
        max = np.diag(ipmat).max()
        for (i,j), val in np.ndenumerate(ipmat):
            if i==j:
                ret[i,j] = val/max
            else:
                v = val/np.sqrt(ipmat[i,i])/np.sqrt(ipmat[j,j])
                ret[i,j] = v
                ret[j,i] = v
        return ret 

    def normalized_ipmatrix(self, vectors):
        norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
        v = vectors / norms.reshape(-1, 1)
        return np.inner(v,v)

    def assign_edge_labels(self, vertex):
        """Edge assignment is geometry dependent. This will try to 
        find the best assignment based on inner product comparison
        with the non-placed lattice arcs."""
        sbu = self._vertex_sbu[vertex]
        print vertex, sbu.name
        local_arcs = sbu.connect_points
        edges = self._net.graph.outgoing_edges(vertex) + \
                    self._net.graph.incoming_edges(vertex)
        indices = self._net.return_indices(edges)
        lattice_arcs = self._net.lattice_arcs
        e_assign = {}
        vects = [self.vector_from_cp_SBU(cp, sbu) for cp in local_arcs]
        li = self.normalized_ipmatrix(vects)
        min=15000.
        cc, assign = None, None
        #print "new batch"
        cell = Cell()
        cell.mkcell(self._net.get_3d_params())
        lattice_vects = np.dot(lattice_arcs, cell.lattice)
        for e in itertools.permutations(edges):
            indices = self._net.return_indices(e)
            node_arcs = lattice_arcs[indices]*\
                    self._net.metric_tensor*lattice_arcs[indices].T
            max = node_arcs.max()
            la = np.empty((len(indices),len(indices)))
            for (i,j), val in np.ndenumerate(node_arcs):
                if i==j:
                    la[i,j] = val/max
                else:
                    v = val/np.sqrt(node_arcs[i,i])/np.sqrt(node_arcs[j,j])
                    la[i,j] = v
                    la[j,i] = v
            # using tensor product of the incidences
            coeff = np.array([-1. if j in self._net.graph.incoming_edges(vertex)
                               else 1. for j in e])
            td = np.tensordot(coeff, coeff, axes=0)
            mm = np.sum(np.absolute(np.multiply(li, td) - la))
            # NB Chirality matters!!!
            # get the cell
            self.chiral_match(e, lattice_vects[indices], sbu)
            # get the lattice arcs
            if (mm < min):# and \
                    #self.chiral_match(e, lattice_vects[indices], sbu):
                cc = coeff
                min = mm
                assign = e
        # NB special MULT function for connect points
        cp_vert = [i[0] if i[0] != vertex else i[1] for i in assign]
        sbu.edge_assignments = assign
        for cp, v in zip(local_arcs, cp_vert):
            cp.vertex_assign = v
        return {e[2]:cp for (e,cp) in zip(assign, local_arcs)}


    def chiral_match(self, edges, arcs, sbu):
        """Determines if two geometries match in terms of edge
        orientation.

        DOI:10.1098/rsif.2010.0297
        """
        edge_weights = [float(e[2][1:]) for e in edges]
        # just rank in terms of weights.......
        edge_weights = [float(sorted(edge_weights).index(e)+1) for e in edge_weights]
        cp_vects = np.array([self.vector_from_cp_SBU(cp, sbu) for cp
                             in sbu.connect_points])

        norms = np.apply_along_axis(np.linalg.norm, 1, arcs.T)
        arcs = arcs.T / norms.reshape(-1, 1)

        norms = np.apply_along_axis(np.linalg.norm, 1, cp_vects.T)
        cp_vects = cp_vects.T / norms.reshape(-1, 1)

        cprm = raw_moment(edge_weights, cp_vects.T)
        com = cprm(0,0,0)
        (mx, my, mz) = (cprm(1,0,0)/com, 
                        cprm(0,1,0)/com, 
                        cprm(0,0,1)/com)
        cpcm = central_moment(edge_weights, cp_vects.T, (mx, my, mz))

        CI_cm = get_CI(cpcm)
        arrm = raw_moment(edge_weights, np.array(arcs).T)
        com = arrm(0,0,0)
        (mx, my, mz) = (arrm(1,0,0)/com, 
                        arrm(0,1,0)/com, 
                        arrm(0,0,1)/com)
        arcm = central_moment(edge_weights, np.array(arcs).T, (mx, my, mz))
        CI_ar = get_CI(arcm)
        print 'elipsoid volumes', elipsoid_vol(cpcm), elipsoid_vol(arcm)
        print 'sbu CI', CI_cm, 'arc CI', CI_ar
        return all(item >= 0 for item in (CI_ar, CI_cm)) or all(item < 0 for item in (CI_ar, CI_cm))

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
        # In cases where there are more than 3 edges, assignment can 
        # get tricky.
        g = self._net.graph
        self._inner_product_matrix = np.zeros((self.net.shape, self.net.shape))
        for v in self.sbu_vertices:
            allvects = {}
            self.assign_edge_labels(v)
            sbu = self._vertex_sbu[v]
            sbu_edges = sbu.edge_assignments
            cps = sbu.connect_points
            vectors = [self.vector_from_cp_SBU(cp, sbu) for cp in cps]
            for i, ed in enumerate(sbu_edges):
                if ed in g.incoming_edges(v):
                    vectors[i]*=-1.

            allvects = {e:vec for e, vec in zip(sbu_edges, vectors)}
            for cp in cps:
                cpv = cp.vertex_assign
                cpe = g.outgoing_edges(cpv) + g.incoming_edges(cpv)
                assert len(cpe) == 2
                edge = cpe[0] if cpe[0] not in sbu_edges else cpe[1]
                vectr = self.vector_from_cp(cp)
                vectr = -1.*vectr if edge in g.incoming_edges(cpv) else vectr
                allvects.update({edge:vectr})

            for (e1, e2) in itertools.combinations_with_replacement(allvects.keys(), 2):
                (i1, i2) = self._net.return_indices([e1, e2])
                dp = np.dot(allvects[e1], allvects[e2])
                self._inner_product_matrix[i1, i2] = dp
                self._inner_product_matrix[i2, i1] = dp
        self._inner_product_matrix = np.asmatrix(self._inner_product_matrix)
        
    def net_degrees(self):
        n = self._net.original_graph.degree_histogram()
        return sorted([i for i, j in enumerate(n) if j])

    def obtain_embedding(self):
        """Optimize the edges and cell parameters to obtain the crystal
        structure embedding.

        """
        # We first need to normalize the edge lengths of the net. This will be
        # done initially by setting the longest vector equal to the longest
        # vector of the barycentric embedding.
        self._net.assign_ip_matrix(np.matrix(self._inner_product_matrix))

        # this calls the optimization routine to match the tensor product matrix
        # of the SBUs and the net.
        self._net.get_embedding()
        test = np.array([0.5, 0.5, 0.5])
        self.build_structure_from_net(test)
        #self.show()

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
        #self.show()

    def build_structure_from_net(self, init_placement):
        """Orient SBUs to the nodes on the net, create bonds where needed, etc.."""
        struct = Structure(self.options, name="test", params=self._net.get_3d_params())
        cell = struct.cell.lattice
        V = self.net.graph.vertices()[0] 
        edges = self.net.graph.outgoing_edges(V) + self.net.graph.incoming_edges(V)
        sbu_pos = self._net.vertex_positions(edges, [], pos={V:init_placement})
        for v in self.sbu_vertices:
            self.sbu_orient(v, cell)
            fc = sbu_pos[v]
            tv = np.dot(fc, cell)
            self.sbu_translate(v, tv)
            struct.add_sbu(self._vertex_sbu[v])

        struct.write_cif()

    def rotation_function(self, params, sbu_vects, data):
        #axis = np.array((params['a1'].value, params['a2'].value, params['a3'].value))
        #angle = params['angle'].value
        #R = rotation_matrix(axis, angle)
        omega = np.array([params['w1'].value, params['w2'].value, params['w3'].value])
        R = rotation_from_omega(omega)
        res = np.dot(R[:3,:3], sbu_vects.T)
        norms = np.apply_along_axis(np.linalg.norm, 1, res.T)
        v = res.T / norms.reshape(-1, 1)
        or1 = np.zeros(3)
        or2 = np.array([3., 3., 0.])
        xyz_str1 = "C %9.5f %9.5f %9.5f\n"%(or1[0], or1[1], or1[2])
        xyz_str2 = "C %9.5f %9.5f %9.5f\n"%(or2[0], or2[1], or2[2])
        for ind, (i, j) in enumerate(zip(v, data)):
            if ind == 0:
                at = "H"
            elif ind == 1:
                at = "F"
            elif ind == 2:
                at = "He"
            elif ind == 3:
                at = "X"
            pos = i + or1
            xyz_str1 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])
            pos = j + or2
            xyz_str2 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2]) 

        xyz_file = open("debugging.xyz", 'a')
        xyz_file.writelines("%i\ndebug\n"%(len(v)*2+2))
        xyz_file.writelines(xyz_str1)
        xyz_file.writelines(xyz_str2)
        xyz_file.close()
        angles = np.array([calc_angle(v1, v2) for v1, v2 in zip(v, data)])
        return angles
        #return (v - data).flatten()

    def sbu_orient(self, v, cell):
        """Optimize the rotation to match vectors"""
        sbu = self._vertex_sbu[v]
        g = self._net.graph
        debug("Orienting SBU: %i, %s on vertex %s"%(sbu.identifier, sbu.name, v))
        # re-index the edges to match the order of the connect points in the sbu list
        indexed_edges = sbu.edge_assignments
        coefficients = np.array([-1. if e in g.outgoing_edges(v) else 1. for e in indexed_edges])
        if len(indexed_edges) != sbu.degree:
            error("There was an error assigning edges "+
                        "to the sbu %s"%(sbu.name))
            Terminate(errcode=1)
        inds = self._net.return_indices(indexed_edges)
        arcs = np.dot(self._net.lattice_arcs[inds], cell)
        norms = np.apply_along_axis(np.linalg.norm, 1, arcs)
        # get the right orientation of the arcs (all pointing away from the node)
        # **********************MAY BREAK STUFF
        arcs = np.array(arcs / norms.reshape(-1, 1)) * coefficients[:, None]
        # **********************MAY BREAK STUFF 
        sbu_vects = np.array([self.vector_from_cp_SBU(cp, sbu) 
                                for cp in sbu.connect_points])
        norms = np.apply_along_axis(np.linalg.norm, 1, sbu_vects)
        sbu_vects = np.array(sbu_vects / norms.reshape(-1, 1))
        #print np.inner(arcs, arcs)
        #print np.inner(sbu_vects, sbu_vects)
        # Try quaternion??
        params = Parameters()
        #params.add('a1', value=0.001, min=-1., max=1.)
        #params.add('a2', value=0.001, min=-1., max=1.)
        #params.add('a3', value=0.001, min=-1., max=1.)
        ## make sure that the angle range covers all 3d rotations...
        #params.add('angle', value=np.pi/2., min=0., max=np.pi)
        params.add('w1', value=1.000)
        params.add('w2', value=1.000)
        params.add('w3', value=1.000)
        min = Minimizer(self.rotation_function, params, fcn_args=(sbu_vects, arcs))
        # giving me a hard time
        #min.lbfgsb(factr=100., epsilon=0.001, pgtol=0.001)
        #print report_errors(params)
        #min = minimize(self.rotation_function, params, args=(sbu_vects, arcs), method='anneal')
        min.leastsq(xtol=1.e-8, ftol=1.e-7)
        #min.fmin()
        #axis = np.array([params['a1'].value, params['a2'].value, params['a3'].value])
        #angle = params['angle'].value
        R = rotation_from_omega(np.array([params['w1'].value, params['w2'].value, params['w3'].value]))
        self.report_errors(sbu_vects, arcs, rot_mat=R)
        #R = rotation_matrix(axis, angle)
        sbu.rotate(R)

    #def sbu_orient(self, v, cell):
    #    """Least squares optimization of orientation matrix.
    #    Obtained from:
    #    Soderkvist & Wedin
    #    'Determining the movements of the skeleton using well configured markers'
    #    J. Biomech. 26, 12, 1993, 1473-1477.
    #    DOI: 10.1016/0021-9290(93)90098-Y"""
    #    g = self._net.graph
    #    sbu = self._vertex_sbu[v]
    #    edges = g.outgoing_edges(v) + g.incoming_edges(v)
    #    debug("Orienting SBU: %i, %s on vertex %s"%(sbu.identifier, sbu.name, v))
    #    # re-index the edges to match the order of the connect points in the sbu list
    #    indexed_edges = sbu.edge_assignments
    #    coefficients = np.array([-1. if e in g.outgoing_edges(v) else 1. for e in indexed_edges])
    #    if len(indexed_edges) != sbu.degree:
    #        error("There was an error assigning edges "+
    #                    "to the sbu %s"%(sbu.name))
    #        Terminate(errcode=1)
    #    inds = self._net.return_indices(indexed_edges)
    #    arcs = np.dot(np.array(self._net.lattice_arcs[inds]), cell)

    #    norms = np.apply_along_axis(np.linalg.norm, 1, arcs)
    #    # get the right orientation of the arcs (all pointing away from the node)
    #    # **********************MAY BREAK STUFF
    #    arcs = np.array(arcs / norms.reshape(-1, 1)) * coefficients[:,None]
    #    # **********************MAY BREAK STUFF 
    #    sbu_vects = np.array([self.vector_from_cp_SBU(cp, sbu) 
    #                            for cp in sbu.connect_points])
    #    norms = np.apply_along_axis(np.linalg.norm, 1, sbu_vects)
    #    sbu_vects = sbu_vects / norms.reshape(-1, 1)
    #    #print np.dot(arcs, arcs.T)
    #    #sf = self._net.scale_factor
    #    #la = self._net.lattice_arcs
    #    #mt = self._net.metric_tensor/sf
    #    #obj = la*mt*la.T
    #    #print obj
    #    #sys.exit()

    #    R = rotation_from_vectors(sbu_vects, arcs) 
    #    self.report_errors(sbu_vects, arcs, rot_mat=R)
    #    sbu.rotate(R)

    def report_errors(self, sbu_vects, arcs, rot_mat=None, axis=None, angle=None):
        if rot_mat is None:
            rot_mat = rotation_matrix(axis, angle)
        rotation = np.dot(rot_mat[:3,:3], sbu_vects.T)
        norms = np.apply_along_axis(np.linalg.norm, 1, rotation.T)
        v = rotation.T / norms.reshape(-1, 1)
        angles = np.array([calc_angle(v1, v2) for v1, v2 in zip(v, arcs)])
        mean, std = np.mean(angles), np.std(angles)
        debug("Average orientation error: %12.6f +/- %9.6f degrees"%(mean/DEG2RAD, std/DEG2RAD))

    def sbu_translate(self, v, trans):
        sbu = self._vertex_sbu[v]
        sbu.translate(trans)

    def show(self):
        g = GraphPlot(self._net)
        #g.view_graph()
        g.view_placement(init=(0.2, 0.2, 0.3))

    def vector_from_cp_SBU(self, cp, sbu):
        for atom in sbu.atoms:
            # NB: THIS BREAKS BARIUM MOFS!!
            for b in atom.sbu_bridge:
                if b == cp.identifier:
                    coords = atom.coordinates[:3]
                    break
        return coords - sbu.COM[:3]

    def vector_from_cp(self,cp):
        return cp.z[:3]/np.linalg.norm(cp.z[:3]) * self.options.sbu_bond_length
        #return cp.origin[:3].copy()# + cp.z[:3]

    @property
    def check_net(self):
        if self._net.original_graph.size() < 25 and self.sbu_degrees == self.net_degrees():
            return True
        return False

    @property
    def sbu_degrees(self):
        if self._sbu_degrees is not None:
            return self._sbu_degrees
        else:
            deg = [i.degree for i in self._sbus]
            lin = [i.linear for i in self._sbus]
            # added a 'set' here in case two different SBUs have the same
            # coordination number
            self._sbu_degrees = sorted(set([j for i,j in zip(lin,deg) if not i]))
            return self._sbu_degrees

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, (graph, volt)):
        self._net = Net(graph)
        self._net.voltage = volt
        #print self._net.graph.to_undirected().automorphism_group()
        #print self._net.graph.vertices()
        #print self._net.graph.edges()
        # keep track of the sbu vertices
        self.sbu_vertices = self._net.graph.vertices()

        #####
        self._obtain_cycle_bases()
        self._net.barycentric_embedding()
        self.show()
        sys.exit()
        #####
        for e in self._net.graph.edges():
            if e in self._net.graph.loop_edges():
                vertices = self._net.add_edges_between(e, 5)
                # add the middle vertex to the SBU vertices..
                # this is probably not a universal thing.
                self.sbu_vertices.append(vertices[2])
            else:
                self._net.add_edges_between(e, 2)
        #self._net.graph.show(edge_labels=True)
        #raw_input("p\n")
        self._obtain_cycle_bases()
        # start off with the barycentric embedding
        self._net.barycentric_embedding()
        self.show()

    @property
    def sbus(self):
        return self._sbus

    @sbus.setter
    def sbus(self, sbus):
        self._sbus = sbus

