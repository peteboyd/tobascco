#!/usr/bin/env sage-python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from sage.all import *

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
    
    def view_graph(self):
        self.net.show()
        raw_input("Wait for Xwindow, then press any key...\n")

    def view_placement(self, init=np.zeros(3)):

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

