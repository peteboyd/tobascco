#!/usr/bin/env python
import numpy as np
import itertools
from scipy.spatial import distance
from LinAlg import LinAlg, RAD2DEG
from element_properties import Radii
from CIFer import CIF

class Structure(object):
    """Structure class contains atom info for MOF."""
    
    def __init__(self, options, name=None):
        self.name = name
        self.options = options
        self.cell = Cell()
        self.atoms = []
        self.bonds = {}
        self.fragments = [] 
        self.build_directives = None
        self.charge = 0

    def from_build(self, build_obj):
        """Build structure up from the builder object"""
        # sort out the connectivity information
        # copy over all the Atoms
        self.cell = build_obj.periodic_vectors
        index_count = 0
        for order, sbu in enumerate(build_obj.sbus):
            sbu.update_atoms(index_count, order)
            self.charge += sbu.charge
            self.fragments.append((sbu.name, order))
            self.atoms += sbu.atoms
            if any([i in self.bonds.keys() for i in sbu.bonds.keys()]):
                warning("Two bonds with the same indices found when forming"+
                        " the bonding table for the structure!")
            self.bonds.update(sbu.bonds)
            index_count += len(sbu)

        for id, sbu in enumerate(build_obj.sbus):
            for cp in sbu.connect_points:
                sbu2 = build_obj.sbus[cp.sbu_bond[0]]
                cp2_id = cp.sbu_bond[1]
                self.compute_inter_sbu_bonding(sbu, cp.identifier, sbu2, cp2_id)

    def compute_inter_sbu_bonding(self, sbu1, cp1_id, sbu2, cp2_id):
        # find out which atoms are involved in inter-sbu bonding
        atoms1 = [i for i in sbu1.atoms if cp1_id in i.sbu_bridge]
        atoms2 = [j for j in sbu2.atoms if cp2_id in j.sbu_bridge]
        # measure minimum distances between atoms to get the 
        # correct bonding.
        base_atoms = atoms1 if len(atoms1) >= len(atoms2) else atoms2
        bond_atoms = atoms2 if len(atoms2) <= len(atoms1) else atoms1
        for atom in base_atoms:
            if bond_atoms:
                shifted_coords = self.min_img(atom, bond_atoms)
                dist = distance.cdist([atom.coordinates[:3]], shifted_coords)
                dist = dist[0].tolist()
                bond_atom = bond_atoms[dist.index(min(dist))]
                self.bonds.update({tuple(sorted((atom.index, 
                                           bond_atom.index))): "S"})

    def _compute_bond_info(self):
        """Update bonds to contain bond type, distances, and min img
        shift."""
        supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        unit_repr = np.array([5,5,5], dtype=int)
        for (at1, at2), val in self.bonds.items():
            atom1 = self.atoms[at1]
            atom2 = self.atoms[at2]
            fcoords = atom2.scaled_pos(self.cell.inverse) + supercells
            coords = []
            for j in fcoords:
                coords.append(np.dot(j, self.cell.lattice))
            coords = np.array(coords)
            dists = distance.cdist([atom1.coordinates[:3]], coords)
            dists = dists[0].tolist()
            image = dists.index(min(dists))
            dist = min(dists)
            sym = '.' if all([i==0 for i in supercells[image]]) else \
                    "1_%i%i%i"%(tuple(np.array(supercells[image],dtype=int) +
                                      unit_repr))
            self.bonds[(at1, at2)] = (val, dist, sym) 

    def compute_overlap(self):
        """Determines if there is atomistic overlap. Includes periodic
        boundary considerations."""
        for id, atom in enumerate(self.atoms):
            elem1 = atom.element
            non_bonded = [i for i in self.atoms[id:] if 
                    tuple(sorted((atom.index, i.index))) not in self.bonds.keys()]
            indices = [i.index for i in non_bonded] 
            shifted_vectors = self.min_img(atom, non_bonded)
            dist_mat = distance.cdist([atom.coordinates[:3]], shifted_vectors)
            for (atom1, atom2), dist in np.ndenumerate(dist_mat):
                id2 = indices[atom2]
                elem2 = self.atoms[id2].element
                if (Radii[elem1] + Radii[elem2])*self.options.overlap_tolerance > dist:
                    return True
        return False

    def min_img(self, atom, atoms):
        """Orient all atoms to within the minimum image 
        of the provided atom."""
        sc_atom = atom.scaled_pos(self.cell.inverse)
        shifted_coords = []
        for at in atoms:
            scaled = at.scaled_pos(self.cell.inverse)
            shift = np.around(sc_atom - scaled)
            shifted_coords.append(np.dot((scaled+shift), self.cell.lattice))
        return shifted_coords

    def detect_symmetry(self):
        pass

    def re_orient(self):
        """Adjusts cell vectors to lie in the standard directions."""
        frac_coords = [i.in_cell_scaled(self.cell.inverse) 
                        for i in self.atoms]
        self.cell.reorient_lattice()
        for id, atom in enumerate(self.atoms):
            atom.coordinates[:3] = np.dot(frac_coords[id],
                                          self.cell.lattice)

    def write_cif(self):
        """Write structure information to a cif file."""
        self._compute_bond_info()
        c = CIF(name=self.name)
        c.insert_block_order("fragment", 4)
        labels = []
        # data block
        c.add_data("data", data_=self.name)
        c.add_data("data", _audit_creation_date=
                            CIF.label(c.get_time()))
        c.add_data("data", _audit_creation_method=
                            CIF.label("Genstruct v.%4.3f"%(
                                    self.options.version)))
        if self.charge:
            c.add_data("data", _chemical_properties_physical=
                               "net charge is %i"%(self.charge))

        # sym block
        c.add_data("sym", _symmetry_space_group_name_H_M=
                            CIF.label("P1"))
        c.add_data("sym", _symmetry_Int_Tables_number=
                            CIF.label("1"))
        c.add_data("sym", _symmetry_cell_setting=
                            CIF.label("triclinic"))

        # sym loop block
        c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                            CIF.label("'x, y, z'"))

        # cell block
        c.add_data("cell", _cell_length_a=CIF.cell_length_a(self.cell.a))
        c.add_data("cell", _cell_length_b=CIF.cell_length_b(self.cell.b))
        c.add_data("cell", _cell_length_c=CIF.cell_length_c(self.cell.c))
        c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(self.cell.alpha))
        c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(self.cell.beta))
        c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(self.cell.gamma))

        for name, order in self.fragments:
            c.add_data("fragment", _chemical_identifier=CIF.label(order),
                                   _chemical_name=CIF.label(name))
        # atom block
        element_counter = {}
        for atom in self.atoms:
            label = c.get_element_label(atom.element)
            labels.append(label)
            c.add_data("atoms", _atom_site_label=
                                    CIF.atom_site_label(label))
            c.add_data("atoms", _atom_site_type_symbol=
                                    CIF.atom_site_type_symbol(atom.element))
            c.add_data("atoms", _atom_site_description=
                                    CIF.atom_site_description(atom.force_field_type))
            c.add_data("atoms", _atom_site_fragment=CIF.atom_site_fragment(atom.sbu_order))
            fc = atom.scaled_pos(self.cell.inverse)
            c.add_data("atoms", _atom_site_fract_x=
                                    CIF.atom_site_fract_x(fc[0]))
            c.add_data("atoms", _atom_site_fract_y=
                                    CIF.atom_site_fract_y(fc[1]))
            c.add_data("atoms", _atom_site_fract_z=
                                    CIF.atom_site_fract_z(fc[2]))

        # bond block
        for (at1, at2), (type, dist, sym) in self.bonds.items():
            label1 = labels[at1]
            label2 = labels[at2]
            c.add_data("bonds", _geom_bond_atom_site_label_1=
                                        CIF.geom_bond_atom_site_label_1(label1))
            c.add_data("bonds", _geom_bond_atom_site_label_2=
                                        CIF.geom_bond_atom_site_label_2(label2))
            c.add_data("bonds", _geom_bond_distance=
                                        CIF.geom_bond_distance(dist))
            c.add_data("bonds", _geom_bond_site_symmetry_2=
                                        CIF.geom_bond_site_symmetry_2(sym))
            c.add_data("bonds", _ccdc_geom_bond_type=
                                        CIF.ccdc_geom_bond_type(type))

        file = open("%s.cif"%self.name, "w")
        file.writelines(str(c))
        file.close()

class Cell(object):
    """contains periodic vectors for the structure."""
    
    def __init__(self):
        self.basis = 0
        self.lattice = np.identity(3)
        self.nlattice = np.zeros((3,3))
        
    @property
    def inverse(self):
        try:
            return self._ilattice
        except AttributeError:
            self._ilattice = np.array(np.matrix(self.lattice).I)
            return self._ilattice

    def add(self, index, vector):
        """Adds a periodic vector to the lattice."""
        self.lattice[index][:] = vector.copy()
        self.nlattice[index][:] = vector.copy() / np.linalg.norm(vector)
        
    def to_xyz(self):
        """Returns a list of the strings"""
        lines = []
        for vector in self.lattice:
            lines.append("atom_vector %12.5f %12.5f %12.5f\n"%(tuple(vector)))
                         
        return lines

    def __mkparam(self):
        """Update the parameters to match the cell."""
        self._params = np.zeros(6)
        # cell lengths
        self._params[0:3] = [np.linalg.norm(i) for i in self.lattice][:]
        # angles in rad
        self._params[3:6] = [LinAlg.calc_angle(i, j) for i, j in
                            reversed(list(itertools.combinations(self.lattice, 2)))]

    def reorient_lattice(self):
        self.__mkparam()
        a, b, c = self._params[:3]
        al, be, ga = self._params[3:]
        cos_be = np.cos(be)
        cos_ga = np.cos(ga)
        sin_ga = np.sin(ga)
        cos_al = np.cos(al)
        c_x = c*cos_be
        c_y = c*(cos_al - cos_ga*cos_be)/sin_ga
        c_z = np.sqrt(c**2 - c_x**2 - c_y**2)
        self.lattice = np.array([[a, 0., 0.],
                                [b*cos_ga, b*sin_ga, 0.],
                                [c_x, c_y, c_z]])
        del self._ilattice # re-compute the inverse

    @property
    def a(self):
        """Magnitude of cell a vector."""
        return self._params[0]

    @property
    def b(self):
        """Magnitude of cell b vector."""
        return self._params[1]

    @property
    def c(self):
        """Magnitude of cell c vector."""
        return self._params[2]

    @property
    def alpha(self):
        """Cell angle alpha."""
        return self._params[3]*RAD2DEG

    @property
    def beta(self):
        """Cell angle beta."""
        return self._params[4]*RAD2DEG

    @property
    def gamma(self):
        """Cell angle gamma."""
        return self._params[5]*RAD2DEG

