# -*- coding: utf-8 -*-
import itertools
import math
import sys
from copy import deepcopy
from functools import reduce
from logging import debug, error, info, warning

import numpy as np
from networkx import degree_histogram

from .config import Terminate
from .linalg import (
    DEG2RAD,
    calc_angle,
    central_moment,
    elipsoid_vol,
    get_CI,
    normalized_vectors,
    raw_moment,
    rotation_from_omega,
    rotation_from_vectors,
)
from .net import Net
from .sbu import SBU
from .structure import Cell, Structure
from .visualizer import GraphPlot


class Build(object):
    """Build a MOF from SBUs and a Net."""

    def __init__(self, options=None):
        self.name = ""
        self._net = None
        self.options = options
        self._sbus = []
        self.scale = 1.0
        self._vertex_sbu = {}
        self._edge_assign = {}
        self._sbu_degrees = None
        self._inner_product_matrix = None
        self.success = False
        self.embedded_net = None

    def _obtain_cycle_bases(self):
        self._net.simple_cycle_basis()
        i = self._net.get_lattice_basis()
        if i < 0:
            return i
        # self._net.get_cycle_basis()
        self._net.get_cocycle_basis()
        return i

    def fit_function(self, params, data):
        val = np.zeros(len(params))
        for p in params:
            ind = int(p.split("_")[1])
            val[ind] = data[ind][int(p.value)]
        return val

    def assign_vertices(self):
        """Assign SBUs to particular vertices in the graph"""
        # TODO(pboyd): assign sbus intelligently, based on edge lengths
        # and independent cycles in the net... ugh
        # set up SBU possibilities on each node,
        # score them based on their orientation
        # score based on alternate organic/metal SBU
        # If > 2 SBUs: automorphism of vertices?

        # get number of unique SBUs
        # get number of nodes which support each SBU

        # get geometric match for each node with each SBU
        # metal - organic bond priority (only if metal SBU and organic SBU have same incidence)
        # assign any obvious ones: ie. if there is only one match between sbu and vertex valency
        temp_vertices = self.sbu_vertices[:]
        for vert in self.sbu_vertices:
            vert_deg = self._net.graph.degree(vert)

            sbu_match = [i for i in self._sbus if i.degree == vert_deg]
            if len(sbu_match) == 1:
                bu = deepcopy(sbu_match[0])
                temp_vertices.pop(temp_vertices.index(vert))
                self._vertex_sbu[vert] = bu
                bu.vertex_id = vert
                [cp.set_sbu_vertex(bu.vertex_id) for cp in bu.connect_points]
        # the remaining we will need to choose (automorphisms?)
        # orbits = self._net.graph.automorphism_group(orbits=True)[1]
        init_verts = []
        for vert in temp_vertices[:]:
            neighbours = self._net.original_graph.neighbors(vert)
            neighbour_sbus = [i for i in neighbours if i in self._vertex_sbu.keys()]
            if neighbour_sbus:
                init_verts.append(vert)
                temp_vertices.pop(temp_vertices.index(vert))
        # re-order so the vertices with neighbours already assigned will be assigned
        # new SBUs first
        temp_vertices = init_verts + temp_vertices
        for vert in temp_vertices[:]:
            # For DiGraphs in networkx, neighbors function only returns the
            # successors of this vertex. Add predecessors to get full function
            neighbours = self._net.original_graph.neighbors(
                vert
            ) + self._net.original_graph.predecessors(vert)

            vert_deg = self._net.graph.degree(vert)

            sbu_match = [i for i in self._sbus if i.degree == vert_deg]
            neighbour_sbus = [i for i in neighbours if i in self._vertex_sbu.keys()]
            temp_assign = []
            for sbu in sbu_match:
                # decide to assign if neighbours are of opposite 'type'
                good_by_type = True
                for neighbour in neighbour_sbus:
                    if sbu.is_metal == self._vertex_sbu[neighbour].is_metal:
                        good_by_type = False
                if good_by_type:
                    temp_assign.append(sbu)

            if len(temp_assign) == 1:
                bu = deepcopy(temp_assign[0])
                temp_vertices.pop(temp_vertices.index(vert))
                self._vertex_sbu[vert] = bu
                bu.vertex_id = vert
                [cp.set_sbu_vertex(bu.vertex_id) for cp in bu.connect_points]
            elif len(temp_assign) > 1:
                self._vertex_sbu[vert] = self.select_sbu(vert, temp_assign)
                temp_vertices.pop(temp_vertices.index(vert))
                bu = self._vertex_sbu[vert]
                bu.vertex_id = vert
                [cp.set_sbu_vertex(bu.vertex_id) for cp in bu.connect_points]
            else:
                self._vertex_sbu[vert] = self.select_sbu(vert, sbu_match)
                temp_vertices.pop(temp_vertices.index(vert))
                bu = self._vertex_sbu[vert]
                bu.vertex_id = vert
                [cp.set_sbu_vertex(bu.vertex_id) for cp in bu.connect_points]
        # for vert in self.sbu_vertices:
        #    # is there a way to determine the symmetry operations applicable
        #    # to a vertex?
        #    # if so, we could compare with SBUs...
        #    vert_deg = self._net.graph.degree(vert)
        #    sbu_match = [i for i in self._sbus if i.degree == vert_deg]
        #    # match tensor product matrices
        #    if len(sbu_match) > 1:
        #        self._vertex_sbu[vert] = self.select_sbu(vert, sbu_match)
        #        bu = self._vertex_sbu[vert]
        #    elif len(sbu_match) == 0:
        #        error("Didn't assign an SBU to vertex %s"%vert)
        #        Terminate(errcode=1)
        #    else:
        #        self._vertex_sbu[vert] = deepcopy(sbu_match[0])
        #        bu = self._vertex_sbu[vert]
        #    bu.vertex_id = vert
        #    [cp.set_sbu_vertex(bu.vertex_id) for cp in bu.connect_points]

        # check to ensure all sbus were assigned to vertices.
        collect = [(sbu.name, sbu.identifier) for vert, sbu in self._vertex_sbu.items()]
        if len(set(collect)) < len(self._sbus):
            remain = [
                s for s in self._sbus if (s.name, s.identifier) not in set(collect)
            ]
            closest_matches = [self.closest_match_vertices(sbu) for sbu in remain]
            taken_verts = []
            for id, bu in enumerate(remain):
                cm = closest_matches[id]
                inds = np.where([np.allclose(x[0], cm[0][0], atol=0.1) for x in cm])
                replace_verts = [
                    i[1] for i in np.array(cm)[inds] if i[1] not in taken_verts
                ]
                taken_verts += replace_verts
                for v in replace_verts:
                    bb = deepcopy(bu)
                    bb.vertex_id = v
                    [cp.set_sbu_vertex(v) for cp in bb.connect_points]
                    self._vertex_sbu[v] = bb

    def closest_match_vertices(self, sbu):
        g = self._net.graph

        if sbu.two_connected and not sbu.linear:
            cp_v = normalized_vectors(
                [
                    self.vector_from_cp_intersecting_pt(cp, sbu)
                    for cp in sbu.connect_points
                ]
            )
        else:
            cp_v = normalized_vectors(
                [self.vector_from_cp_SBU(cp, sbu) for cp in sbu.connect_points]
            )
        ipv = self.scaled_ipmatrix(np.inner(cp_v, cp_v))

        inds = np.triu_indices(ipv.shape[0], k=1)
        max, min = np.absolute(ipv[inds]).max(), np.absolute(ipv[inds]).min()
        cmatch = []
        for v in self.sbu_vertices:
            ee = self._net.neighbours(v)
            l_arcs = self._net.lattice_arcs[self._net.return_indices(ee)]
            lai = np.dot(np.dot(l_arcs, self._net.metric_tensor), l_arcs.T)
            ipc = self.scaled_ipmatrix(lai)
            imax, imin = np.absolute(ipc[inds]).max(), np.absolute(ipc[inds]).min()
            mm = np.sum(np.absolute([max - imax, min - imin]))
            cmatch.append((mm, v))
        return sorted(cmatch)

    def select_sbu(self, v, sbus):
        """This is a hackneyed way of selecting the right SBU,
        will use until it breaks something.

        """
        edges = self._net.neighbours(v)
        indices = self._net.return_indices(edges)
        lattice_arcs = self._net.lattice_arcs[indices]
        ipv = np.dot(np.dot(lattice_arcs, self._net.metric_tensor), lattice_arcs.T)
        ipv = self.scaled_ipmatrix(ipv)
        # just take the max and min angles...
        inds = np.triu_indices(ipv.shape[0], k=1)
        max, min = np.absolute(ipv[inds]).max(), np.absolute(ipv[inds]).min()
        minmag = 15000.0
        # There is likely many problems with this method. Need to ensure that there
        # are enough nodes to assign the breadth of SBUs used to build the MOF.
        sbus_assigned = [i.name for i in self._vertex_sbu.values()]
        neighbours_assigned = {}
        for nn in self.sbu_joins[v]:
            try:
                nnsbu = self._vertex_sbu[nn]
                # neighbours_assigned[nn] = (nnsbu.name, nnsbu.is_metal)
                neighbours_assigned[nn] = nnsbu.is_metal
            except KeyError:
                neighbours_assigned[nn] = None

        not_added = [i.name for i in sbus if i.name not in sbus_assigned]

        for sbu in sbus:
            if sbu.two_connected and not sbu.linear:
                vects = np.array(
                    [
                        self.vector_from_cp_intersecting_pt(cp, sbu)
                        for cp in sbu.connect_points
                    ]
                )
            else:
                vects = np.array(
                    [self.vector_from_cp_SBU(cp, sbu) for cp in sbu.connect_points]
                )
            ipc = self.scaled_ipmatrix(np.inner(vects, vects))
            imax, imin = np.absolute(ipc[inds]).max(), np.absolute(ipc[inds]).min()
            mm = np.sum(np.absolute([max - imax, min - imin]))
            if any([sbu.is_metal == nn for nn in neighbours_assigned.values()]):
                mm *= 2.0
            elif sbu.name in not_added:
                mm = 0.0

            if mm < minmag:
                minmag = mm
                assign = sbu
        return deepcopy(assign)

    def obtain_edge_vector_from_cp(self, cp1):
        """Create an edge vector from an sbu's connect point"""
        e1 = self.vector_from_cp(cp1)
        len1 = np.linalg.norm(e1[:3])
        dir = e1[:3] / len1
        return dir * self.options.sbu_bond_length

    def scaled_ipmatrix(self, ipmat):
        """Like normalized inner product matrix, however the
        diagonal is scaled to the longest vector."""
        ret = np.empty_like(ipmat)
        max = np.diag(ipmat).max()
        for (i, j), val in np.ndenumerate(ipmat):
            if i == j:
                ret[i, j] = val / max
            if i != j:
                v = val / np.sqrt(ipmat[i, i]) / np.sqrt(ipmat[j, j])
                ret[i, j] = v
                ret[j, i] = v
        return ret

    def normalized_ipmatrix(self, vectors):
        v = normalized_vectors(vectors)
        return np.inner(v, v)

    def assign_edge_labels(self, vertex):
        """Edge assignment is geometry dependent. This will try to
        find the best assignment based on inner product comparison
        with the non-placed lattice arcs."""
        sbu = self._vertex_sbu[vertex]
        local_arcs = sbu.connect_points
        edges = self._net.neighbours(vertex)
        indices = self._net.return_indices(edges)
        lattice_arcs = self._net.lattice_arcs
        e_assign = {}
        if sbu.two_connected and not sbu.linear:
            vects = [self.vector_from_cp_intersecting_pt(cp, sbu) for cp in local_arcs]
        else:
            vects = [self.vector_from_cp_SBU(cp, sbu) for cp in local_arcs]
        norm_cp = normalized_vectors(vects)
        li = self.normalized_ipmatrix(vects)
        min, chi_diff = 15000.0, 15000.0
        cc, assign = None, None
        debug("%s assigned to %s" % (sbu.name, vertex))
        cell = Cell()
        cell.mkcell(self._net.get_3d_params())
        if self._net.ndim == 2:
            lattice_arcs = np.hstack(
                (lattice_arcs, np.zeros((lattice_arcs.shape[0], 1)))
            )
        lattice_vects = np.dot(lattice_arcs, cell.lattice)
        count = 0
        for e in itertools.permutations(edges):
            count += 1
            indices = self._net.return_indices(e)
            # node_arcs = lattice_arcs[indices]*\
            #        self._net.metric_tensor*lattice_arcs[indices].T
            # max = node_arcs.max()
            # la = np.empty((len(indices),len(indices)))
            # for (i,j), val in np.ndenumerate(node_arcs):
            #    if i==j:
            #        la[i,j] = val/max
            #    else:
            #        v = val/np.sqrt(node_arcs[i,i])/np.sqrt(node_arcs[j,j])
            #        la[i,j] = v
            #        la[j,i] = v
            # using tensor product of the incidences
            coeff = np.array(
                [-1.0 if j in self._net.in_edges(vertex) else 1.0 for j in e]
            )
            # td = np.tensordot(coeff, coeff, axes=0)
            # diff = np.multiply(li, td) - la
            # inds = np.triu_indices(diff.shape[0], k=1)
            # xmax, xmin = np.absolute(diff[inds]).max(), np.absolute(diff[inds]).min()
            # mm = np.sum(diff)
            # mm = np.sum(np.absolute(np.multiply(li,td) - la))
            # NB Chirality matters!!!
            # get the cell
            lv_arc = np.array(lattice_vects[indices]) * coeff[:, None]
            # get the lattice arcs

            mm = self.get_chiral_diff(e, lv_arc, vects)

            # norm_arc = normalized_vectors(lv_arc)
            # orient the lattice arcs to the first sbu vector...
            # print count , self.chiral_match(e, oriented_arc, norm_cp)
            # print count, np.allclose(norm_cp, oriented_arc, atol=0.01)
            # or1 = np.zeros(3)
            # or2 = np.array([3., 3., 0.])
            # xyz_str1 = "C %9.5f %9.5f %9.5f\n"%(or1[0], or1[1], or1[2])
            # xyz_str2 = "C %9.5f %9.5f %9.5f\n"%(or2[0], or2[1], or2[2])
            # for ind, (i, j) in enumerate(zip(norm_cp,oriented_arc)):
            #    at = atoms[ind]
            #    pos = i[:3] + or1
            #    xyz_str1 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])
            #    pos = j + or2
            #    xyz_str2 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])

            # xyz_file = open("debugging.xyz", 'a')
            # xyz_file.writelines("%i\ndebug\n"%(len(norm_cp)*2+2))
            # xyz_file.writelines(xyz_str1)
            # xyz_file.writelines(xyz_str2)
            # xyz_file.close()

            # print "arc CI", CI_ar, "cp  CI", CI_cp
            # if (mm < min) and (diff < chi_diff):
            if mm <= min:  # and self.chiral_match(e, norm_arc, norm_cp):#, tol=xmax):
                min = mm
                assign = e
        # CI = self.chiral_invariant(assign, norm_arc)
        # axis = np.array([1., 3., 1.])
        # angle = np.pi/3.
        # R = rotation_matrix(axis, angle)
        # new_norm = np.dot(R[:3,:3], norm_arc.T)
        # nCI = self.chiral_invariant(assign, new_norm.T)
        # print "Rotation invariant?", CI, nCI
        # NB special MULT function for connect points
        cp_vert = [i[0] if i[0] != vertex else i[1] for i in assign]
        # print 'CI diff', chi_diff
        # print 'tensor diff', mm
        sbu.edge_assignments = assign
        for cp, v in zip(local_arcs, cp_vert):
            cp.vertex_assign = v

    def chiral_invariant(self, edges, vectors):
        edge_weights = [float(e[2][1:]) for e in edges]
        # just rank in terms of weights.......
        edge_weights = [float(sorted(edge_weights).index(e) + 1) for e in edge_weights]
        vrm = raw_moment(edge_weights, vectors)
        com = vrm(0, 0, 0)
        (mx, my, mz) = (vrm(1, 0, 0) / com, vrm(0, 1, 0) / com, vrm(0, 0, 1) / com)
        vcm = central_moment(edge_weights, vectors, (mx, my, mz))
        return get_CI(vcm)

    def get_chiral_diff(self, edges, arc1, arc2, count=[]):
        narcs1 = normalized_vectors(arc1)
        narcs2 = normalized_vectors(arc2)
        ### DEBUG
        # atoms = ["H", "F", "He", "Cl", "N", "O"]
        R = rotation_from_vectors(narcs2, narcs1)
        # FIXME(pboyd): ensure that this is the right rotation!!! I think it's supposed to rotate narcs2
        narcs1 = (np.dot(R[:3, :3], narcs1.T)).T
        # narcs2 = (np.dot(R[:3,:3], narcs2.T)).T
        # or1 = np.zeros(3)
        # or2 = np.array([3., 3., 0.])
        # xyz_str1 = "C %9.5f %9.5f %9.5f\n"%(or1[0], or1[1], or1[2])
        # xyz_str2 = "C %9.5f %9.5f %9.5f\n"%(or2[0], or2[1], or2[2])
        # for ind, (i, j) in enumerate(zip(narcs1,narcs2)):
        #    at = atoms[ind]
        #    pos = i[:3] + or1
        #    xyz_str1 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])
        #    pos = j + or2
        #    xyz_str2 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])

        # xyz_file = open("debugging.xyz", 'a')
        # xyz_file.writelines("%i\ndebug\n"%(len(narcs1)*2+2))
        # xyz_file.writelines(xyz_str1)
        # xyz_file.writelines(xyz_str2)
        # xyz_file.close()

        ### DEBUG
        # CI_1 = self.chiral_invariant(edges, narcs1)

        # CI_2 = self.chiral_invariant(edges, narcs2)
        # count.append(1)
        # ff = open("CI1", 'a')
        # ff.writelines('%i %e\n'%(len(count), CI_1))
        # ff.close()
        # ff = open("CI2", 'a')
        # ff.writelines('%i %e\n'%(len(count), CI_2))
        # ff.close()
        # print 'edge assignment ', ','.join([p[2] for p in edges])
        # print 'lattice arcs  CI ', CI_1
        # print 'connect point CI ', CI_2

        # if all(item >= 0 for item in (CI_1, CI_2)) or all(item < 0 for item in (CI_1, CI_2)):
        #    return np.absolute(CI_1 - CI_2)
        # else:
        #    return 150000.
        return np.sum(np.absolute((narcs1 - narcs2).flatten()))

    def chiral_match(self, edges, arcs, cp_vects, tol=0.01):
        """Determines if two geometries match in terms of edge
        orientation.

        DOI:10.1098/rsif.2010.0297
        """
        edge_weights = [float(e[2][1:]) for e in edges]
        # just rank in terms of weights.......
        edge_weights = [float(sorted(edge_weights).index(e) + 1) for e in edge_weights]
        vrm = raw_moment(edge_weights, cp_vects)
        com = vrm(0, 0, 0)
        (mx, my, mz) = (vrm(1, 0, 0) / com, vrm(0, 1, 0) / com, vrm(0, 0, 1) / com)
        vcm = central_moment(edge_weights, cp_vects, (mx, my, mz))
        if np.allclose(elipsoid_vol(vcm), 0.0, atol=0.004):
            return True
        # This is a real hack way to match vectors...
        R = rotation_from_vectors(arcs[:], cp_vects[:])
        oriented_arc = (np.dot(R[:3, :3], arcs.T)).T
        return np.allclose(cp_vects, oriented_arc, atol=tol)

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
        # In cases where there is asymmetry in the SBU or the vertex, assignment can
        # get tricky.
        g = self._net.graph
        self._inner_product_matrix = np.zeros((self.net.shape, self.net.shape))
        self.colattice_inds = ([], [])
        for v in self.sbu_vertices:
            allvects = {}
            self.assign_edge_labels(v)
            sbu = self._vertex_sbu[v]
            sbu_edges = sbu.edge_assignments
            cps = sbu.connect_points
            if sbu.two_connected and not sbu.linear:
                vectors = [self.vector_from_cp_intersecting_pt(cp, sbu) for cp in cps]
            else:
                vectors = [self.vector_from_cp_SBU(cp, sbu) for cp in cps]
            for i, ed in enumerate(sbu_edges):
                if ed in self._net.in_edges(v):
                    vectors[i] *= -1.0

            allvects = {e: vec for e, vec in zip(sbu_edges, vectors)}
            for cp in cps:
                cpv = cp.vertex_assign
                cpe = self._net.neighbours(cpv)
                assert len(cpe) == 2
                edge = cpe[0] if cpe[0] not in sbu_edges else cpe[1]
                # temporarily set to the vertex of the other connect point
                cp.bonded_cp_vertex = edge[0] if edge[0] != cpv else edge[1]
                vectr = self.obtain_edge_vector_from_cp(cp)
                vectr = -1.0 * vectr if edge in self._net.in_edges(cpv) else vectr
                allvects.update({edge: vectr})

            for (e1, e2) in itertools.combinations_with_replacement(allvects.keys(), 2):
                (i1, i2) = self._net.return_indices([e1, e2])
                dp = np.dot(allvects[e1], allvects[e2])
                self.colattice_inds[0].append(i1)
                self.colattice_inds[1].append(i2)
                self._inner_product_matrix[i1, i2] = dp
                self._inner_product_matrix[i2, i1] = dp
        self._inner_product_matrix = np.asmatrix(self._inner_product_matrix)

    def net_degrees(self):
        # n = self._net.original_graph.to_undirected().degree_histogram() # SAGE compatible
        n = degree_histogram(self._net.original_graph)  # networkx compatible
        return sorted([i for i, j in enumerate(n) if j])

    def obtain_embedding(self):
        """Optimize the edges and cell parameters to obtain the crystal
        structure embedding.

        """
        # We first need to normalize the edge lengths of the net. This will be
        # done initially by setting the longest vector equal to the longest
        # vector of the barycentric embedding.
        self._net.assign_ip_matrix(self._inner_product_matrix, self.colattice_inds)

        # this calls the optimization routine to match the tensor product matrix
        # of the SBUs and the net.
        optimized = True
        optimized = self._net.net_embedding()

        init = np.array([0.5, 0.5, 0.5])
        if self.bad_embedding() or not optimized:
            warning(
                "net %s didn't embed properly with the " % (self._net.name)
                + "geometries dictated by the SBUs"
            )
        else:
            self.build_structure_from_net(init)

    def test_angle(self, index1, index2, mat):
        return (
            np.arccos(
                mat[index1, index2]
                / np.sqrt(mat[index1, index1])
                / np.sqrt(mat[index2, index2])
            )
            * 180.0
            / np.pi
        )

    def custom_embedding(self, rep, mt):
        self._net.metric_tensor = np.matrix(mt)
        self._net.periodic_rep = np.matrix(rep)
        la = np.dot(self._net.cycle_cocycle.I, rep)
        ip = np.dot(np.dot(la, mt), la.T)
        ipsbu = self._inner_product_matrix
        nz = np.nonzero(np.triu(ipsbu))
        self.build_structure_from_net(np.zeros(self._net.ndim))
        # self.show()

    def bad_embedding(self):
        mt = self._net.metric_tensor
        lengths = []
        angles = []
        for (i, j) in zip(*np.triu_indices_from(mt)):
            if i == j:
                if mt[i, j] <= 0.0:
                    warning("The cell lengths reported were less than zero!")
                    return True
                lengths.append(np.sqrt(mt[i, j]))
            else:
                dp_mag = mt[i, j] / mt[i, i] / mt[j, j]
                try:
                    angles.append(math.acos(dp_mag))
                except ValueError:
                    warning("The cell angles reported were less than zero!")
                    return True

        vol = np.sqrt(np.linalg.det(mt))
        if vol < self.options.cell_vol_tolerance * reduce(lambda x, y: x * y, lengths):
            warning("The unit cell obtained was below the specified volume tolerance")
            return True
        # v x w = ||v|| ||w|| sin(t)
        return False

    def build_structure_from_net(self, init_placement):
        """Orient SBUs to the nodes on the net, create bonds where needed, etc.."""
        metals = "_".join(
            ["m%i" % (sbu.identifier) for sbu in self._sbus if sbu.is_metal]
        )
        organics = "_".join(
            ["o%i" % (sbu.identifier) for sbu in self._sbus if not sbu.is_metal]
        )
        name = "str_%s_%s_%s" % (metals, organics, self._net.name)
        self.name = name
        # name += "_ftol_%11.5e"%self.options.ftol
        # name += "_xtol_%11.5e"%self.options.xtol
        # name += "_eps_%11.5e"%self.options.epsfcn
        # name += "_fac_%6.1f"%self.options.factor
        struct = Structure(self.options, name=name, params=self._net.get_3d_params())

        cell = struct.cell.lattice
        V = self.net.vertices(0)
        edges = self.net.neighbours(V)
        sbu_pos = self._net.vertex_positions(edges, [], pos={V: init_placement})
        for v in self.sbu_vertices:
            self.sbu_orient(v, cell)
            fc = sbu_pos[v]
            tv = np.dot(fc, cell)
            self.sbu_translate(v, tv)
            # compute dihedral angle, if one exists...
            struct.add_sbu(self._vertex_sbu[v])
        struct.connect_sbus(self._vertex_sbu)
        if self.options.overlap_tolerance != 0.0 and struct.compute_overlap():
            warning("Overlap found in final structure, not creating MOF.")
        else:
            struct.write_cif()
            self.struct = struct
            self.success = True
            if self.options.store_net:
                self.embedded_net = self.store_placement(cell, init_placement)
            info("Structure Generated!")

    def rotation_function(self, params, vect1, vect2):
        # axis = np.array((params['a1'].value, params['a2'].value, params['a3'].value))
        # angle = params['angle'].value
        # R = rotation_matrix(axis, angle)
        omega = np.array([params["w1"].value, params["w2"].value, params["w3"].value])
        R = rotation_from_omega(omega)
        res = np.dot(R[:3, :3], vect1.T).T

        # v = normalized_vectors(res.T)
        ### DEBUGGGGGG
        # or1 = np.zeros(3)
        # or2 = np.array([3., 3., 0.])
        # xyz_str1 = "C %9.5f %9.5f %9.5f\n"%(or1[0], or1[1], or1[2])
        # xyz_str2 = "C %9.5f %9.5f %9.5f\n"%(or2[0], or2[1], or2[2])
        # atms = ["H", "F", "O", "He", "N", "Cl"]
        # for ind, (i, j) in enumerate(zip(res, vect2)):
        #    at = atms[ind]
        #    pos = i + or1
        #    xyz_str1 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])
        #    pos = j + or2
        #    xyz_str2 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])

        # xyz_file = open("debug_rotation_function.xyz", 'a')
        # xyz_file.writelines("%i\ndebug\n"%(len(res)*2+2))
        # xyz_file.writelines(xyz_str1)
        # xyz_file.writelines(xyz_str2)
        # xyz_file.close()
        ### DEBUGGGGGG
        # angles = np.array([calc_angle(v1, v2) for v1, v2 in zip(v, vect2)])
        # return angles
        return (res - vect2).flatten()

    def get_rotation_matrix(self, vect1, vect2):
        """Optimization to match vectors, obtain rotation matrix to rotate
        vect1 to vect2"""
        params = Parameters()
        params.add("w1", value=1.000)
        params.add("w2", value=1.000)
        params.add("w3", value=1.000)
        min = Minimizer(self.rotation_function, params, fcn_args=(vect1, vect2))
        # giving me a hard time
        min.lbfgsb(factr=100.0, epsilon=0.001, pgtol=0.001)
        # print report_errors(params)
        # min = minimize(self.rotation_function, params, args=(sbu_vects, arcs), method='anneal')
        # min.leastsq(xtol=1.e-8, ftol=1.e-7)
        # min.fmin()
        # axis = np.array([params['a1'].value, params['a2'].value, params['a3'].value])
        # angle = params['angle'].value
        R = rotation_from_omega(
            np.array([params["w1"].value, params["w2"].value, params["w3"].value])
        )
        return R

    # def sbu_orient(self, v, cell):
    #    """Optimize the rotation to match vectors"""
    #    sbu = self._vertex_sbu[v]
    #    g = self._net.graph
    #    debug("Orienting SBU: %i, %s on vertex %s"%(sbu.identifier, sbu.name, v))
    #    # re-index the edges to match the order of the connect points in the sbu list
    #    indexed_edges = sbu.edge_assignments
    #    coefficients = np.array([1. if e in g.outgoing_edges(v) else -1. for e in indexed_edges])
    #    if len(indexed_edges) != sbu.degree:
    #        error("There was an error assigning edges "+
    #                    "to the sbu %s"%(sbu.name))
    #        Terminate(errcode=1)
    #    inds = self._net.return_indices(indexed_edges)
    #    arcs = np.dot(self._net.lattice_arcs[inds], cell)

    #    # get the right orientation of the arcs (all pointing away from the node)
    #    # **********************MAY BREAK STUFF
    #    arcs = normalized_vectors(arcs) * coefficients[:, None]
    #    # **********************MAY BREAK STUFF
    #    sbu_vects = np.array([self.vector_from_cp_SBU(cp, sbu)
    #                            for cp in sbu.connect_points])

    #    sbu_vects = normalized_vectors(sbu_vects)
    #    #print np.inner(arcs, arcs)
    #    #print np.inner(sbu_vects, sbu_vects)
    #    # Try quaternion??
    #    params = Parameters()
    #    #params.add('a1', value=0.001, min=-1., max=1.)
    #    #params.add('a2', value=0.001, min=-1., max=1.)
    #    #params.add('a3', value=0.001, min=-1., max=1.)
    #    ## make sure that the angle range covers all 3d rotations...
    #    #params.add('angle', value=np.pi/2., min=0., max=np.pi)
    #    params.add('w1', value=1.000)
    #    params.add('w2', value=1.000)
    #    params.add('w3', value=1.000)
    #    min = Minimizer(self.rotation_function, params, fcn_args=(sbu_vects, arcs))
    #    # giving me a hard time
    #    #min.lbfgsb(factr=10., epsilon=0.00001, pgtol=0.000001)
    #    #print report_errors(params)
    #    #min = minimize(self.rotation_function, params, args=(sbu_vects, arcs), method='anneal')
    #    min.leastsq(xtol=1.e-8, ftol=1.e-7)
    #    #min.lbfgsb(factr=1000., approx_grad=True, m=200, epsilon=1e-6, pgtol=1e-6)
    #    #min.fmin()
    #    #axis = np.array([params['a1'].value, params['a2'].value, params['a3'].value])
    #    #angle = params['angle'].value
    #    R = rotation_from_omega(np.array([params['w1'].value, params['w2'].value, params['w3'].value]))
    #    self.report_errors(sbu_vects, arcs, R)
    #    #R = rotation_matrix(axis, angle)
    #    sbu.rotate(R)

    def sbu_orient(self, v, cell):
        """Least squares optimization of orientation matrix.
        Obtained from:
        Soderkvist & Wedin
        'Determining the movements of the skeleton using well configured markers'
        J. Biomech. 26, 12, 1993, 1473-1477.
        DOI: 10.1016/0021-9290(93)90098-Y"""
        g = self._net.graph
        sbu = self._vertex_sbu[v]
        edges = self._net.neighbours(v)
        debug("Orienting SBU: %i, %s on vertex %s" % (sbu.identifier, sbu.name, v))
        # re-index the edges to match the order of the connect points in the sbu list
        indexed_edges = sbu.edge_assignments
        coefficients = np.array(
            [1.0 if e in self._net.out_edges(v) else -1.0 for e in indexed_edges]
        )
        if len(indexed_edges) != sbu.degree:
            error("There was an error assigning edges " + "to the sbu %s" % (sbu.name))
            Terminate(errcode=1)

        inds = self._net.return_indices(indexed_edges)
        la = self._net.lattice_arcs[inds]
        if self._net.ndim == 2:
            la = np.hstack((la, np.zeros((la.shape[0], 1))))

        arcs = np.dot(la, cell)
        arcs = normalized_vectors(arcs) * coefficients[:, None]

        sbu_vects = normalized_vectors(
            np.array([self.vector_from_cp_SBU(cp, sbu) for cp in sbu.connect_points])
        )

        # print np.dot(arcs, arcs.T)
        # sf = self._net.scale_factor
        # la = self._net.lattice_arcs
        # mt = self._net.metric_tensor/sf
        # obj = la*mt*la.T
        # print obj
        # issue for ditopic SBUs where the inner product matrices could invert the
        # angles (particularly for ZIFs)
        if sbu.degree == 2 and not sbu.linear:
            sbu_vects = normalized_vectors(
                np.array(
                    [
                        self.vector_from_cp_intersecting_pt(cp, sbu)
                        for cp in sbu.connect_points
                    ]
                )
            )
            # define the plane generated by the edges
            # print "arc angle %9.5f"%(180.*calc_angle(*arcs)/np.pi)
            # print "sbu angle %9.5f"%(180.*calc_angle(*sbu_vects)/np.pi)
            # For some reason the least squares rotation matrix
            # does not work well with just two vectors, so a third
            # orthonormal vector is included to create the proper
            # rotation matrix
            arc3 = np.cross(arcs[0], arcs[1])
            arc3 /= np.linalg.norm(arc3)
            cp3 = np.cross(sbu_vects[0], sbu_vects[1])
            cp3 /= np.linalg.norm(cp3)
            sbu_vects = np.vstack((sbu_vects, cp3))
            arcs = np.vstack((arcs, arc3))
        R = rotation_from_vectors(sbu_vects, arcs)
        mean, std = self.report_errors(sbu_vects, arcs, rot_mat=R)

        ### DEBUGGGGGG
        # or1 = np.zeros(3)
        # or2 = np.array([3., 3., 0.])
        # xyz_str1 = "C %9.5f %9.5f %9.5f\n"%(or1[0], or1[1], or1[2])
        # xyz_str2 = "C %9.5f %9.5f %9.5f\n"%(or2[0], or2[1], or2[2])
        # atms = ["H", "F", "O", "He", "N", "Cl"]
        # sbu_rot_vects = np.dot(R[:3,:3], sbu_vects.T)
        # for ind, (i, j) in enumerate(zip(arcs, sbu_rot_vects.T)):
        #    at = atms[ind]
        #    pos = i + or1
        #    xyz_str1 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])
        #    pos = j + or2
        #    xyz_str2 += "%s %9.5f %9.5f %9.5f\n"%(at, pos[0], pos[1], pos[2])

        # xyz_file = open("debug_rotation_function.xyz", 'a')
        # xyz_file.writelines("%i\ndebug\n"%(len(sbu_rot_vects.T)*2+2))
        # xyz_file.writelines(xyz_str1)
        # xyz_file.writelines(xyz_str2)
        # xyz_file.close()
        ### DEBUGGGGGG

        mean, std = self.report_errors(sbu_vects, arcs, rot_mat=R)
        debug(
            "Average orientation error: %12.6f +/- %9.6f degrees"
            % (mean / DEG2RAD, std / DEG2RAD)
        )
        sbu.rotate(R)

    def report_errors(self, sbu_vects, arcs, rot_mat):
        rotation = np.dot(rot_mat[:3, :3], sbu_vects.T)
        v = normalized_vectors(rotation.T)
        angles = np.array([calc_angle(v1, v2) for v1, v2 in zip(v, arcs)])
        mean, std = np.mean(angles), np.std(angles)
        return mean, std

    def sbu_translate(self, v, trans):
        sbu = self._vertex_sbu[v]
        sbu.translate(trans)

    def show(self):
        g = GraphPlot(self._net)
        # g.view_graph()
        sbu_verts = self.sbu_vertices
        g.view_placement(init=(0.51, 0.51, 0.51), edge_labels=False, sbu_only=sbu_verts)

    def vector_from_cp_intersecting_pt(self, cp, sbu):
        for atom in sbu.atoms:
            # NB: THIS BREAKS BARIUM MOFS!!
            coords = cp.origin[:3]
            for b in atom.sbu_bridge:
                if b == cp.identifier:
                    coords = atom.coordinates[:3]
                    break
        v = coords - sbu.closest_midpoint
        return v

    def vector_from_cp_SBU(self, cp, sbu):
        # coords = cp.origin[:3]
        for atom in sbu.atoms:
            # NB: THIS BREAKS BARIUM MOFS!!
            for b in atom.sbu_bridge:
                if b == cp.identifier:
                    coords = atom.coordinates[:3]
                    break
        # fix for tetrahedral metal ions
        if np.allclose(coords - sbu.COM[:3], np.zeros(3)):
            return cp.origin[:3] - coords
        return coords - sbu.COM[:3]

    def vector_from_cp(self, cp):
        return cp.z[:3] / np.linalg.norm(cp.z[:3])
        # return cp.origin[:3].copy()# + cp.z[:3]

    @property
    def check_net(self):
        # if self._net.original_graph.size() < 25 and self.sbu_degrees == self.net_degrees():
        min = self.options.min_edge_count
        max = self.options.max_edge_count
        if (
            self.sbu_degrees == self.net_degrees()
            and self._net.original_graph.size() <= max
            and self._net.original_graph.size() >= min
        ):
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
            # added incidence != 2 for nonlinear sbus.
            self._sbu_degrees = sorted(
                set([j for i, j in zip(lin, deg) if not i and j != 2])
            )
            return self._sbu_degrees

    @property
    def linear_sbus(self):
        """Return true if one or more of the SBUs are linear"""
        # Not sure why this is limited to only linear SBUs.
        # should apply to any ditopic SBUs, hopefully the program will
        # sort out the net shape during the optimization.
        for s in self._sbus:
            if s.degree == 2:  # and s.linear:
                return True
        return False

    @property
    def met_met_bonds(self):
        met_incidence = [sbu.degree for sbu in self._sbus if sbu.is_metal]
        org_incidence = [sbu.degree for sbu in self._sbus if not sbu.is_metal]

        # if org and metal have same incidences, then just ignore this...
        # NB may still add met-met bonds in the net!
        if set(met_incidence).intersection(set(org_incidence)):
            return False
        # for (v1, v2, e) in self._net.graph.edges(): # SAGE compliant
        for (v1, v2, e) in self._net.neighbours(None):
            nn1 = len(self._net.neighbours(v1))
            nn2 = len(self._net.neighbours(v2))

            if (nn1 in met_incidence) or (nn2 in met_incidence):
                if (nn1 == nn2) or ((v1, v2, e) in self.net.loop_edges()):
                    return True

        return False

    def init_embed(self):
        # keep track of the sbu vertices
        edges_split = []
        self.sbu_vertices = list(self._net.vertices())
        met_incidence = [sbu.degree for sbu in self._sbus if sbu.is_metal]
        org_incidence = [sbu.degree for sbu in self._sbus if not sbu.is_metal]
        # Some special cases: linear sbus and no loops.
        # Insert between metal-type vertices
        # if self.linear_sbus and not self._net.graph.loop_edges():
        #    for (v1, v2, e) in self._net.graph.edges():
        #        nn1 = len(self._net.neighbours(v1))
        #        nn2 = len(self._net.neighbours(v2))
        #        if nn1 == nn2 and (nn1 in met_incidence):
        #            vertices, edges = self._net.add_edges_between((v1, v2, e), 5)
        #            self.sbu_vertices.append(vertices[2])
        #            edges_split += edges
        self.sbu_joins = {}
        for (v1, v2, e) in self._net.all_edges():
            if (v1, v2, e) not in edges_split:
                nn1 = len(self._net.neighbours(v1))
                nn2 = len(self._net.neighbours(v2))
                # LOADS of ands here.
                if self.linear_sbus:
                    if ((v1, v2, e) in self._net.loop_edges()) or (
                        (nn1 == nn2) and (nn1 in met_incidence)
                    ):
                        vertices, edges = self._net.add_edges_between((v1, v2, e), 5)
                        # add the middle vertex to the SBU vertices..
                        # this is probably not a universal thing.
                        self.sbu_joins.setdefault(vertices[2], []).append(v1)
                        self.sbu_joins.setdefault(v1, []).append(vertices[2])
                        self.sbu_joins.setdefault(vertices[2], []).append(v2)
                        self.sbu_joins.setdefault(v2, []).append(vertices[2])
                        self.sbu_vertices.append(vertices[2])

                        edges_split += edges
                    else:
                        self.sbu_joins.setdefault(v1, []).append(v2)
                        self.sbu_joins.setdefault(v2, []).append(v1)
                        vertices, edges = self._net.add_edges_between((v1, v2, e), 2)
                        edges_split += edges
                else:
                    self.sbu_joins.setdefault(v1, []).append(v2)
                    self.sbu_joins.setdefault(v2, []).append(v1)
                    vertices, edges = self._net.add_edges_between((v1, v2, e), 2)
                    edges_split += edges

        i = self._obtain_cycle_bases()
        if i < 0:
            return i
        # start off with the barycentric embedding
        self._net.barycentric_embedding()
        return i

    def setnet(self, tupl):
        (name, graph, volt) = tupl
        dim = volt.shape[1]
        self._net = Net(graph, dim=dim, options=self.options)
        self._net.name = name
        self._net.voltage = volt

    def getnet(self):
        return self._net

    net = property(getnet, setnet)

    def getsbus(self):
        return self._sbus

    def setsbus(self, sbus):
        self._sbus = sbus

    sbus = property(getsbus, setsbus)

    def get_automorphisms(self):
        """Compute the automorphisms associated with the graph.
        Automorphisms are defined as a permutation, s, of the vertex set such that a
        pair of vertices, (u,v) form an edge if and only if the pair (s(u),s(v)) also
        form an edge. I have to identify all the edge swappings according to the
        permutation groups presented by sage. I need to define all the edge permutations,
        then I can identify the symmetry operations associated with these permutations."""

        G = self.net.original_graph.to_undirected().automorphism_group()
        count = 0
        for i in G:
            count += 1
            # find equivalent edges after vertex automorphism

            # construct linear representation

            # determine symmetry element (rotation, reflection, screw, glide, inversion..)

            # chose to discard or keep. Does it support the site symmetries of the SBUs?

            if count == 2:
                break

        # final set of accepted symmetry operations

        # determine space group

        # determine co-lattice vectors which keep symmetry elements intact

        # determine lattice parameters.

        # self.net.original_graph.order()
        # self.net.original_graph.edges()

    def store_placement(self, cell, init=(0.0, 0.0, 0.0)):
        init = np.array(init)
        data = {"cell": cell, "nodes": {}, "edges": {}}
        # set the first node down at the init position
        V = self._net.vertices(0)
        edges = self._net.neigbours(V)
        unit_cell_vertices = self._net.vertex_positions(edges, [], pos={V: init})
        for key, value in unit_cell_vertices.items():
            if key in self._vertex_sbu.keys():
                label = self._vertex_sbu[key].name
            else:
                label = key
                for bu in self._vertex_sbu.values():
                    for cp in bu.connect_points:
                        if cp.vertex_assign == key:
                            label = str(cp.identifier)
            data["nodes"][label] = np.array(value)
            for edge in self._net.out_edges(key):
                ind = self._net.get_index(edge)
                arc = np.array(self._net.lattice_arcs)[ind]
                data["edges"][edge[2]] = (np.array(value), arc)
            for edge in self._net.in_edges(key):
                ind = self._net.get_index(edge)
                arc = -np.array(self._net.lattice_arcs)[ind]
                data["edges"][edge[2]] = (np.array(value), arc)
        return data
