import numpy as np
from element_properties import WEIGHT

class Atom(object):
    """Basic atom class for the generation of structures."""
    
    def __init__(self, element=None):
        null = np.array([0., 0., 0., 1.])
        self.element = element
        # atom index
        self.index = 0
        # the index of the SBU this atom is associated with.
        self.sbu_index = None
        # order of the SBU in which it is placed in the unit cell.
        self.sbu_order = None
        self.sbu_metal = False
        # is this an atom that connects to another SBU?
        self.sbu_bridge = [] 
        self.force_field_type = None
        self.coordinates = null.copy()
        self.neighbours = []
    
    def scaled_pos(self, inv_cell): 
        return np.dot(self.coordinates[:3], inv_cell)

    def in_cell_scaled(self, inv_cell):
        return np.array([i%1 for i in self.scaled_pos(inv_cell)])

    def in_cell(self, cell, inv_cell):
        return np.dot(self.in_cell_scaled(inv_cell), cell)

    @property
    def mass(self):
        return WEIGHT[self.element]
    
    def from_config_ff(self, line):
        """Parse data from old config file format"""
        line = line.strip().split()
        self.element = line[0]
        self.force_field_type = line[1]
        for i, c in enumerate(line[2:]):
            self.coordinates[i] = float(c)
            
    def from_config(self, line):
        """New config file format, just element, x, y, z"""
        line = line.strip().split()
        self.element = line[0]
        for i, c in enumerate(line[1:]):
            self.coordinates[i] = float(c)
    
    def rotate(self, R):
        self.coordinates[:3] = np.dot(R[:3,:3], self.coordinates[:3])
        
    def translate(self, vector):
        self.coordinates[:3] += vector

