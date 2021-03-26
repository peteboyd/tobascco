# -*- coding: utf-8 -*-
import warnings

import numpy as np

__all__ = ["ConnectPoint"]


class ConnectPoint:
    def __init__(self):
        """Origin describes the point of intersection of two parameters,
        z describes the vector pointing along the bond (parallel),
        y describes a vector perpendicular to z for alignment purposes.
        """
        null = np.array([0.0, 0.0, 0.0, 1.0])
        self.identifier = None
        self.origin = np.zeros(4)
        self.y = null.copy()
        self.z = null.copy()
        # flag to determine if the point has been attached
        self.connected = False
        # order which the SBU was placed in the Structure
        self.sbu_vertex = None
        self.bonded_cp_vertex = None
        self.constraint = None
        self.special = None
        self.symmetry = 1
        self.vertex_assign = None

    def set_sbu_vertex(self, val):
        assert self.sbu_vertex is None
        self.sbu_vertex = val

    def from_config(self, line):
        """ Obtain the connectivity information from the config .ini file."""
        line = line.strip().split()
        self.identifier = int(line[0])
        # obtain the coordinate information.
        self.origin[:3] = np.array([float(x) for x in line[1:4]])
        try:
            self.z[:3] = np.array([float(x) for x in line[4:7]])
        except ValueError:
            warnings.warn(
                "Improper formatting of input SBU file! cannot find the"
                + "connecting vector for bond %i." % (self.identifier)
                + "Catastrophic errors in the bonding will ensue!"
            )
        try:
            self.y[:3] = np.array([float(x) for x in line[7:10]])
        except ValueError:
            # Y not needed at the moment.
            pass
        if len(line) == 12:
            try:
                self.symmetry = int(line[10])
            except ValueError:
                self.symmetry = 1
            try:
                self.special = int(line[11])
            except ValueError:
                self.special = None
        self._normalize()

    def _normalize(self):
        """Normalize the y and z vectors"""
        self.z[:3] = self.z[:3] / np.linalg.norm(self.z[:3])
        # self.y[:3] = self.y[:3]/np.linalg.norm(self.y[:3])

    def rotate(self, R):
        self.origin = np.dot(R, self.origin)
        self.y[:3] = np.dot(R[:3, :3], self.y[:3])
        self.z[:3] = np.dot(R[:3, :3], self.z[:3])

    def translate(self, vector):
        self.origin[:3] += vector

    def __mul__(self, val):
        self.origin[:3] *= val
        self.z[:3] *= val
