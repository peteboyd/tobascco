#!/usr/bin/env sage-python

import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sage.all import *
from sage.plot.point import point
from sage.plot.line import line

# try pygraphs cycle iterator
sys.path.append("/home/pboyd/lib/python-graph/core/build/lib")
from pygraph.algorithms.cycles import find_cycle

import numpy as np
from scipy.optimize import fmin, minimize 
np.set_printoptions(threshold=np.nan, precision=4, suppress=True, linewidth=185)

DEG2RAD = np.pi / 180.0

class GraphPlot(object):
    
    def __init__(self, net, two_dimensional=False):
        self.fig, self.ax = plt.subplots()
        self.net = net
        self.two_dimensional = two_dimensional
        if two_dimensional:
            self.cell = np.identity(2)
            self.params = net.get_2d_params()
            self.__mkcell()
            self.plot_2d_cell()
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.params = net.get_3d_params()
            self.cell = np.identity(3)
            self.__mkcell()
            self.plot_3d_cell()

    def plot_2d_cell(self, origin=np.zeros(2), colour='b'):
        xyz_a = (self.cell[0] + origin)/ 2.
        xyz_b = (self.cell[1] + origin)/ 2.
        self.fig.text(xyz_a[0], xyz_a[1], 'a')
        self.fig.text(xyz_b[0], xyz_b[1], 'b')
        all_points = [np.sum(a, axis=0)+origin
                      for a in list(self.powerset(self.cell)) if a]
        all_points.append(origin)
        for s, e in itertools.combinations(np.array(all_points), 2):
            if any([self.zero_cross(s-e, i) for i in self.cell]):
                self.ax.plot(*zip(s, e), color=colour)

    def plot_3d_cell(self, origin=np.zeros(3), colour='b'):

        # add axes labels
        xyz_a = (self.cell[0]+origin)/2.
        xyz_b = (self.cell[1]+origin)/2.
        xyz_c = (self.cell[2]+origin)/2.
        self.ax.text(xyz_a[0], xyz_a[1], xyz_a[2], 'a')
        self.ax.text(xyz_b[0], xyz_b[1], xyz_b[2], 'b')
        self.ax.text(xyz_c[0], xyz_c[1], xyz_c[2], 'c')

        all_points = [np.sum(a, axis=0)+origin
                      for a in list(self.powerset(self.cell)) if a]
        all_points.append(origin)
        for s, e in itertools.combinations(np.array(all_points), 2):
            if any([self.zero_cross(s-e, i) for i in self.cell]):
                #line([tuple(s), tuple(e)], rgbcolor=(0,0,255))
                self.ax.plot3D(*zip(s, e), color=colour)
                
    def add_point(self, p=np.zeros(3), label=None, colour='r'):
        pp = np.dot(p.copy(), self.cell)
        try:
            self.ax.scatter(*pp, color=colour)
        except TypeError:
            pp = pp.tolist()
            self.ax.scatter(pp, color=colour)
        if label:
            #point(tuple(pp), legend_label=label, rgbcolor=(255,0,0))
            self.ax.text(*pp, s=label)

    def add_edge(self, vector, origin=np.zeros(3), label=None, colour='y'):
        """Accounts for periodic boundaries by splitting an edge where
        it intersects with the plane of the boundary conditions.

        """
        p = origin + vector
        p1 = np.dot(origin, self.cell)
        p2 = np.dot(p, self.cell)
        self.ax.plot3D(*zip(p2, p1), color=colour)
        if label:
            pp = (p2 + p1)*0.5
            #pp = pp - np.floor(p)
            #line([tuple(p1), tuple(p2)], rgbcolor=(255,255,0), legend_label=label)
            self.ax.text(*pp, s=label)
        else:
            line([tuple(p1), tuple(p2)], rgbcolor=(255,255,0))
    
    def __mkcell(self):
        """Update the cell representation to match the parameters."""
        if self.two_dimensional:
            a_mag, b_mag = self.params[:2]
            gamma = DEG2RAD * self.params[2]
            a_vec = np.array([a_mag, 0.])
            b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma)])
            self.cell = np.array([a_vec, b_vec])
        else:
            a_mag, b_mag, c_mag = self.params[:3]
            alpha, beta, gamma = [x * DEG2RAD for x in self.params[3:]]
            a_vec = np.array([a_mag, 0.0, 0.0])
            b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma), 0.0])
            c_x = c_mag * np.cos(beta)
            c_y = c_mag * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
            c_vec = np.array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5])
            self.cell = np.array([a_vec, b_vec, c_vec])
            
    def plot(self, init=np.zeros(3)):

        # set the first node down at the init position
        V = self.net.graph.vertices()[0] 
        edges = self.net.graph.outgoing_edges(V) + self.net.graph.incoming_edges(V)
        unit_cell_vertices = self.vertex_positions(edges, [], pos={V:init})
        for key, value in unit_cell_vertices.items():
            self.add_point(p=np.array(value), label=key)
            for edge in self.net.graph.outgoing_edges(key):
                ind = self.net.get_index(edge)
                arc = np.array(self.net.lattice_arcs)[ind]
                self.add_edge(arc, origin=np.array(value), label=edge[2])
        mx = max(self.params[:3])
        self.ax.set_xlim3d(0,mx)
        self.ax.set_ylim3d(0,mx)
        self.ax.set_zlim3d(0,mx)
        plt.show()

    def vertex_positions(self, edges, used, pos={}, bad_ones = {}):
        """Recursive function to find the nodes in the unit cell."""
        if len(pos) == self.net.graph.order() or not edges:
            # check if some of the nodes will naturally fall outside of the 
            # unit cell
            if len(pos) != self.net.graph.order():
                fgtn = set(self.net.graph.vertices()).difference(pos.keys())
                for node in fgtn:
                    poses = [e for e in bad_ones.keys() if node in e[:2]]
                    # just take the first one.. who cares?
                    pos.update({node:bad_ones[poses[0]]})
            return pos
        else:
            e = edges[0]
            if e[0] not in pos.keys() and e[1] not in pos.keys():
                pass
            elif e[0] not in pos.keys() or e[1] not in pos.keys():
                from_v = e[0] if e[0] in pos.keys() else e[1]
                to_v = e[1] if e[1] not in pos.keys() else e[0]
                coeff = 1. if e in self.net.graph.outgoing_edges(from_v) else -1.
                index = self.net.get_index(e)
                to_pos = coeff*np.array(self.net.lattice_arcs)[index] + pos[from_v]
                newedges = []
                if np.all(np.where((to_pos >= -0.00001) & (to_pos < 1.00001), True, False)):
                    pos.update({to_v:to_pos})
                    used.append(e)
                    ee = self.net.graph.outgoing_edges(to_v) + self.net.graph.incoming_edges(to_v)
                    newedges = [i for i in ee if i not in used and i not in edges]
                else:
                    bad_ones.update({e:to_pos})
                edges = newedges + edges[1:]
            else:
                used.append(e)
                edges = edges[1:]
            return self.vertex_positions(edges, used, pos, bad_ones)


    def powerset(self, iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

    def zero_cross(self, vector1, vector2):
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = vector2/np.linalg.norm(vector2)
        return np.allclose(np.zeros(3), np.cross(vector1, vector2), atol=0.01)
    
    def point_of_intersection(self, p_edge, edge, p_plane, plane_vec1, plane_vec2):
        """
        Returns a point of intersection between an edge and a plane
        p_edge is a point on the edge vector
        edge is the vector direction
        p_plane is a point on the plane
        plane_vec1 represents one of the vector directions of the plane
        plane_vec2 represents the second vector of the plane

        """
        n = np.cross(plane_vec1, plane_vec2)
        n = n / np.linalg.norm(n)
        l = edge / np.linalg.norm(edge)
        
        ldotn = np.dot(l, n)
        pdotn = np.dot(p_plane - p_edge, n)
        if ldotn == 0.:
            return np.zeros(3) 
        if pdotn == 0.:
            return p_edge 
        return pdotn/ldotn*l + p_edge 


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
        self.cocycle = np.zeros((size, len))
        for ind, vert in enumerate(self._graph.vertices()[:-1]):
            out_edges = self._graph.outgoing_edges(vert)
            inds = self.return_indices(out_edges)
            if inds:
                self.cocycle[ind][inds] = 1.
            in_edges = self._graph.incoming_edges(vert)
            inds = self.return_indices(in_edges)
            if inds:
                self.cocycle[ind][inds] = -1.
        self.cocycle = np.matrix(self.cocycle)
        self.cocycle_rep = np.matrix(np.zeros((size, self.ndim)))
  
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
        self.cycle = []
        self.cycle_rep = [] 
        
        count = 0
        if self.lattice_basis is not None:
            for id, (cyc, volt) in enumerate(zip(self.lattice_basis, np.identity(self.ndim))):
                self.cycle.append(np.array(cyc))
                self.cycle_rep.append(np.array(volt))
                count += 1
                if count >= n:
                    break

        for id, cycle in enumerate(c):
            if count >= n:
                break
            vect = np.zeros(self.shape)
            vect[self.return_indices(cycle)] = self.return_coeff(cycle)
            volt = self.get_voltage(vect)
            A = np.array(self.cycle + [vect])
            U, s, V = np.linalg.svd(A)
            if np.all(np.abs(volt) < 1.001) and np.all(s > 0.0001):
                self.cycle.append(vect)
                self.cycle_rep.append(volt)
                count += 1
        self.cycle = np.matrix(self.cycle)
        self.cycle_rep = np.matrix(self.cycle_rep)
        del c

    def get_voltage(self, cycle):
        return np.array(cycle*self.voltage)[0]

    def get_cycle_coefficients(self, nodes, cycle):
        """Really gross way of obtaining the proper orientations of the 
        edges for the DiGraph cycle basis.

        """
        edges = self.get_edges_from_index(cycle)
        coefficients, recycle = [], []
        vertices, reverse_flag = [],[]
        for i, j, k in edges:
            vertices.append(i)
            vertices.append(j)
            recycle.append(k)
        vertices = set(vertices)
        for v in vertices:
            out = [i[2] for i in self._graph.outgoing_edges(v)]
            d = list(set(out).intersection(set(recycle)))
            if len(d) == 2:
                reverse_flag.append(d[1])
        for e in edges:
            coeff = 1. if e[2] not in reverse_flag else -1.
            coefficients.append(coeff)

        for ind, i in enumerate(np.nonzero(cycle)[0]):
            cycle[i] = coefficients[ind]
        return cycle 
   
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
                if np.allclose(np.abs(volt), e) and id not in basis_vectors:
                    basis_vectors.append(id)
                    self.lattice_basis[id] = volt[id]*vect
        if len(basis_vectors) != self.ndim:
            print "ERROR: could not find all cycle vectors for the lattice basis!"
            sys.exit()

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

def qtz():
    # qtz net
    qtz = Net()

    qtz.graph = {'A':{'B':['e1','e6']}, 'B':{'C':['e5', 'e2']}, 'C':{'A':['e4', 'e3']}}
    qtz.voltage = np.matrix([[0, 1, 0],
                             [-1, -1, 1],
                             [1, 0, 0],
                             [0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 0]])

    #qtz.graph.show(edge_labels=True)
    qtz.get_lattice_basis()
    qtz.get_cocycle_basis()
    qtz.get_cycle_basis()
    qtz.barycentric_embedding()
    gp = GraphPlot(qtz)
    gp.plot(init=np.array([0., 0., 0.])) 


def bor():
    bor=Net()
    bor.graph = {'A':{'C':['e1'], 'B':['e2'], 'F':['e3']}, 
                 'B':{}, 
                 'C':{}, 
                 'D':{'C':['e7'], 'B':['e9'], 'F':['e11']}, 
                 'E':{'C':['e4'], 'B':['e5'], 'F':['e6']}, 
                 'F':{}, 
                 'G':{'B':['e10'], 'F':['e12'], 'C':['e8']}}
    bor.voltage = np.matrix([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]])
    bor.get_cocycle_basis()
    # NB: lattice basis first because they are included in the cycle basis
    # if not included - then the projection does not work!
    bor.get_lattice_basis()
    bor.get_cycle_basis()
    bor.barycentric_embedding()
    bor_show = GraphPlot(bor)
    bor_show.plot(init=np.array([0.25, 0.25, 0.25]))
    #bor.graph.show(edge_labels=True)
    #raw_input("Type any key to continue...\n")

def alpha_crystobalite():
    acryst = Net()
    acryst.graph = {'A':{'B':['e1','e2'], 'D':['e5','e6']}, 
                    'B':{'C':['e7', 'e8']}, 
                    'C':{'D':['e3', 'e4']},
                    'D':{}}
    #acryst.graph.show(edge_labels=True)
    #raw_input("Press any key...\n")
    acryst.voltage = np.matrix([[0, -1, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1],
                                [-1, 0, -1],
                                [0, 0, 0],
                                [-1, 0, 0]])

    #acryst.graph.show(edge_labels=True)
    acryst.get_cocycle_basis()
    acryst.get_lattice_basis()
    acryst.get_cycle_basis()
    acryst.barycentric_embedding()
    acr_show = GraphPlot(acryst)
    acr_show.plot(init=np.array([0.25, 0.25, 0.]))


def hcb():
    hcb = Net()
    # hcb 2d net
    hcb.lattice_basis = np.matrix([[1., -1., 0.],
                                  [0., 1., -1.]])
    hcb_view = GraphPlot(params=hcb.get_2d_params(), two_dimensional=True)
    hcb_view.plot_2d_cell()
    hcb_view.plot()


def main():
    #hcb()
    #qtz()
    alpha_crystobalite()
    #bor()

if __name__=="__main__":
    main()

