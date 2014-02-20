#!/usr/bin/env python
from Atoms import Atom
from ConnectPoints import ConnectPoint
import ConfigParser
import numpy as np
import itertools
from scipy.spatial import distance
from logging import info, debug, warning, error, critical
from element_properties import Radii
from numpy import pi, cos, sin, arccos

class SBU_list(object):
    
    def __init__(self, sbu_list):
        self.list = sbu_list
        self._truncate()

    def _truncate(self):
        trunc = []
        for sbu1, sbu2 in itertools.combinations(self.list, 2):
            if sbu1.parent == sbu2.name:
                trunc.append(self.list.index(sbu1)) 
                sbu2.children.append(sbu1)
            elif sbu2.parent == sbu1.name:
                trunc.append(self.list.index(sbu2))
                sbu1.children.append(sbu2)
        for k in reversed(sorted(trunc)):
            del self.list[k] 

    def get(self, identifier, _METAL=False):
        """Produces the SBU with the identifier provided, this filters between
        organic and metal SBUs"""
        for sbu in self.list:
            if sbu.identifier == identifier and sbu.is_metal == _METAL:
                return sbu
        raise Exception("Could not find the SBU with the identifier %s"%(identifier))
    
    def getall(self, identifier):
        """Produces the SBU with the target identifier regardless of being
        metal or organic."""
        for sbu in self.list:
            if sbu.identifier == identifier:
                return sbu
        raise Exception("Could not find the SBU with the identifier %s"%(identifier))

class SBU(object):
    """Contains atom information, connectivity of a Secondary Building Unit."""
    
    def __init__(self, name=None):
        self.name = name
        self.identifier = 0
        self.index = 0
        self.topology = None
        self.charge = 0
        self.parent = None
        self.is_metal = False
        self.atoms = []
        # child SBUs which are associated with this one through self.parent
        self.children = []
        self.bonds = {}
        self.connect_points = []
        self.edge_assignments = []
        self.vertex_id = None
        
    def from_config(self, section, cfgdic):
        """take atom and connectivity information from a config file"""
        
        self.name = section
        self.identifier = cfgdic.getint(section, 'index')
        try:
            self.charge = cfgdic.getint(section, 'charge')
        except ConfigParser.NoOptionError:
            # charge not specified in the input file.
            pass
        self.topology = cfgdic.get(section, 'topology')
        self.is_metal = cfgdic.getboolean(section, 'metal')
        if cfgdic.has_option(section, 'parent'):
            self.parent = cfgdic.get(section,'parent')
        # read in atom information
        # depreciated coordinates, but backwards compatible
        if cfgdic.has_option(section, 'coordinates'):
            atom_info = cfgdic.get(section, 'coordinates').strip().splitlines()
        elif cfgdic.has_option(section, 'atoms'):
            atom_info = cfgdic.get(section, 'atoms').strip().splitlines()

        for idx, atom_line in enumerate(atom_info):
            split_atom_line = atom_line.split()
            newatom = Atom()
            newatom.index = idx
            newatom.sbu_index = self.identifier
            newatom.sbu_metal = self.is_metal
            if len(split_atom_line) == 5:
                newatom.from_config_ff(atom_line)
            elif len(split_atom_line) == 4:
                newatom.from_config(atom_line)
            self.atoms.append(newatom)
            
        # bonding table
        if cfgdic.has_option(section, 'table'):
            for table_line in cfgdic.get(section, 'table').strip().splitlines():
                bond = table_line.strip().split()
                # add the bonding information
                # first two cases are for bonding to connecting points
                if "c" in bond[0].lower():
                    connect_ind = int(bond[0].lower().strip('c'))
                    atom_ind = int(bond[1])
                    self.atoms[atom_ind].sbu_bridge.append(connect_ind)
                elif "c" in bond[1].lower():
                    # subtract 1 since the input file starts at 1
                    connect_ind = int(bond[1].lower().strip('c'))
                    atom_ind = int(bond[0])
                    self.atoms[atom_ind].sbu_bridge.append(connect_ind)
                else:
                    b = tuple(sorted([int(bond[0]), int(bond[1])]))
                    self.bonds[b] = bond[2]
                    
        if not self.bonds:
            debug("No bonding found in input file for %s,"%self.name +
                  " so bonding will not be reported")
        # Connect points
        for idx, cp_line in enumerate(cfgdic.get(section, 'connectivity').strip().splitlines()):
            connect_point = ConnectPoint()
            connect_point.from_config(cp_line)
            self.connect_points.append(connect_point)
           
        # check for constraints
        if cfgdic.has_option(section, 'bond_constraints'):
            const_lines = cfgdic.get(section, 'bond_constraints').strip().splitlines()
            for constraint in const_lines:
                constraint = constraint.split()
                id = int(constraint[0])
                con = int(constraint[1])
                cp = self.get_cp(id)
                cp.constraint = con

        # new special/constraint section
        elif cfgdic.has_option(section, 'connect_flag'):
            const_lines = cfgdic.get(section, 'connect_flag').strip().splitlines()
            for constraint in const_lines:
                id, special, const = [int(i) for i in constraint.split()]
                cp = self.get_cp(id)
                cp.special = special
                cp.constraint = const

        # new symmetry flag stuff
        if cfgdic.has_option(section, 'connect_sym'):
            sym_lines = cfgdic.get(section, 'connect_sym').strip().splitlines()
            for sym in sym_lines:
                id, sym_flag = [int(i) for i in sym.split()]
                cp = self.get_cp(id)
                cp.symmetry = sym_flag

    def update_atoms(self, index_base, order):
        self.bonds = {(i+index_base, j+index_base):val for (i, j), val in 
                      self.bonds.items()}
        for atom in self.atoms:
            atom.index += index_base
            atom.sbu_order = order

    def rotate(self, rotation_matrix):
        """Apply the rotation matrix to the coordinates and connect_points in
        the SBU."""
        # rotate the connect points
        [c.rotate(rotation_matrix) for c in self.connect_points]
        
        # rotate the atoms
        [a.rotate(rotation_matrix) for a in self.atoms]
    
    def translate(self, v):
        vector = v - self.COM[:3]
        [c.translate(vector) for c in self.connect_points]
        [i.translate(vector) for i in self.atoms]

    def calc_neighbours(self, radii=None):
        """Determines atom neighbours, based on bonding, and the supplied radii."""
        atom_combos = itertools.combinations(range(len(self.atoms)), 2)
        atom_coordinates = np.array([atom.coordinates[:3] for atom in self.atoms])
        dist_matrix = distance.cdist(atom_coordinates, atom_coordinates)
        for atid1, atid2 in atom_combos:
            atom1 = self.atoms[atid1]
            atom2 = self.atoms[atid2]

            # determine if neighbours
            bid1 = atom1.index
            bid2 = atom2.index
            btest = tuple(sorted([bid1, bid2]))
            # loops over Null if self.bonds = [] (i.e. no bonding info in the input file)
            dist = dist_matrix[atid1, atid2]
            for (i, j), btype in self.bonds.items():
                if btest == (i, j):
                    # append neighbours
                    atom1.neighbours.append((dist, atid2))
                    atom2.neighbours.append((dist, atid1))
            if radii is None:
                # keep a neighbour list of up to 3*(radii) of each atom.
                if dist_matrix[atid1,atid2] < 3.*(Radii[atom1.element] + Radii[atom2.element]):
                    atom1.neighbours.append((dist, atid2))
                    atom2.neighbours.append((dist, atid1))
            else:
                if dist_matrix[atid1,atid2] < radii:
                    atom1.neighbours.append((dist, atid2))
                    atom2.neighbours.append((dist, atid1))
            atom1.neighbours = list(set(atom1.neighbours))
            atom2.neighbours = list(set(atom2.neighbours))
        for atom in self.atoms:
            # sort in order of increasing distance
            atom.neighbours = sorted(atom.neighbours)[:]
           
    @property
    def linear(self):
        """Return true if linear else false"""
        if len(self.connect_points) != 2:
            return False
        if np.allclose(self.connect_points[0].z[:3],-self.connect_points[1].z[:3], atol=1e-3):
            return True
        return False

    @property
    def degree(self):
        return len(self.connect_points)

    @property
    def COM(self):
        return np.average(np.array([atom.coordinates for atom in self.atoms]),axis=0,
                          weights=np.array([atom.mass for atom in self.atoms]))
   
    @property
    def centre_of_atoms(self):
        return np.average(np.array([atom.coordinates for atom in self.atoms]),axis=0)

    @property
    def surface_area(self, probe=1.82, resolution=0.03):
        """Computes surface area. Currently uses default resolution of 0.03 A^2
        and an N2 probe radii of 1.82 A"""
        # make sure we are including neighbours with the correct probe size!
        self.calc_neighbours()
        xyz = []
        surface_area = 0.
        for atom in self.atoms:
            ncount = 0
            radii = Radii[atom.element] + probe
            atom_sa = 4.*pi*(radii**2)
            nsamples = int(atom_sa / resolution)
            phi = np.random.random(nsamples)*pi
            costheta = np.random.random(nsamples)*2. - 1.
            theta = arccos(costheta)
            points = np.array([sin(theta)*cos(phi),
                               sin(theta)*sin(phi),
                               cos(theta)]).transpose()*radii + \
                    atom.coordinates[:3]
            for point in points:
                for dist, atid in atom.neighbours:
                    n_atom = self.atoms[atid]
                    n_radii = Radii[n_atom.element] + probe
                    if dist > radii + n_radii:
                        # neighbours are sorted by distance.
                        # the neighbour atoms are too far apart - include this point.
                        ncount += 1
                        xyz.append((atom.element, point))
                        break
                    elif np.linalg.norm(point - n_atom.coordinates[:3]) < n_radii:
                        # collision with point
                        break
                else:
                    ncount += 1
                    xyz.append((atom.element, point))
                    
            surface_area += (atom_sa*ncount) / nsamples
        return surface_area
    
    @property
    def max_span(self):
        cp_coords = [cp.origin[:3] for cp in self.connect_points]
        dist_matrix = distance.cdist(cp_coords, cp_coords)
        max_dist = 0.
        for cp1, cp2 in itertools.combinations(range(len(self.connect_points)), 2):
            if dist_matrix[cp1, cp2] > max_dist:
                max_dist = dist_matrix[cp1, cp2]
        return max_dist
        
    def get_cp(self, identifier):
        for cp in self.connect_points:
            if identifier == cp.identifier:
                return cp
        error("%i not in the connecting points! "%(identifier)+
              ", ".join([str(i.identifier) for i in self.connect_points]))

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        """Return an .xyz format of the SBU."""
        line = ""
        for cp in self.connect_points:
            if cp.connected:
                atom = "He"
            else:
                atom = "H"
            line += "%s %12.5f %12.5f %12.5f "%(tuple([atom] + cp.origin[:3].tolist()))
            line += "atom_vector %12.5f %12.5f %12.5f "%(tuple(cp.z[:3]))
            line += "atom_vector %12.5f %12.5f %12.5f\n"%(tuple(cp.y[:3]))
            
        for atom in self.atoms:
            line += "%s %12.5f %12.5f %12.5f\n"%(tuple([atom.element] +
                                                 [i for i in atom.coordinates[:3]]))
            
        return line

