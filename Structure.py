#!/usr/bin/env python
import numpy as np
import itertools
import sys
import os
from scipy.spatial import distance
from LinAlg import DEG2RAD, RAD2DEG
from element_properties import Radii, ATOMIC_NUMBER
from CIFer import CIF

class Structure(object):
    """Structure class contains atom info for MOF."""
    
    def __init__(self, options, name=None, **kwargs):
        self.name = name
        self.options = options
        try:
            self.cell = Cell()
            self.cell.mkcell(kwargs['params'])
        except KeyError:
            self.cell = Cell()
        self.atoms = []
        self.bonds = {}
        self.fragments = [] 
        self.build_directives = None
        self.charge = 0
        self.sbu_count = 0
        self.atom_count = 0
        self.space_group_name = 'P1'
        self.space_group_number = 1

    def add_sbu(self, sbu_obj):
        sbu_obj.update_atoms(self.atom_count, self.sbu_count)
        self.charge += sbu_obj.charge
        self.fragments.append((sbu_obj.name, self.sbu_count))
        for atom in sbu_obj.atoms:
            atom.coordinates = atom.in_cell(self.cell.lattice, self.cell.inverse)
        self.atoms += sbu_obj.atoms
        if any([i in self.bonds.keys() for i in sbu_obj.bonds.keys()]):
            warning("Two bonds with the same indices found when forming"+
                    " the bonding table for the structure! Check the final"+
                    " atom bonding to determine any anomalies.")
        self.bonds.update(sbu_obj.bonds)
        self.sbu_count += 1
        self.atom_count += len(sbu_obj)

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

    def connect_sbus(self, sbu_dic):
        cp_pool = {cp.vertex_assign:cp for sbu in sbu_dic.values() for cp in sbu.connect_points}
        for vertex, sbu in sbu_dic.items():
            for cp in sbu.connect_points:
                xx = cp_pool[cp.bonded_cp_vertex]
                bsbu = sbu_dic[xx.sbu_vertex]
                self.compute_inter_sbu_bonding(sbu, cp.identifier, bsbu, xx.identifier)

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
            non_bonded = [i.index for i in self.atoms[id:] if 
                    tuple(sorted((atom.index, i.index))) not in self.bonds.keys()]
            bonded = [i.index for i in self.atoms[id:] if 
                       tuple(sorted((atom.index, i.index))) in self.bonds.keys()]
            indices = [i.index for i in self.atoms[id:]]
            shifted_vectors = self.min_img(atom, self.atoms[id:])
            dist_mat = distance.cdist([atom.coordinates[:3]], shifted_vectors)
            for (atom1, atom2), dist in np.ndenumerate(dist_mat):
                id2 = indices[atom2]
                elem2 = self.atoms[id2].element
                if (id != id2) and (id2 in non_bonded) and \
                    (Radii[elem1] + Radii[elem2])*self.options.overlap_tolerance > dist:
                    return True
                elif (id != id2) and (id2 in bonded) and 1.*self.options.overlap_tolerance > dist:
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
                            CIF.label("TopCryst v.%s"%(
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
        if self.options.find_symmetric_h:
            # find symmetry
            sym = Symmetry(self.options)
            sym.add_structure(self)
            sym.refine_cell()
            h_equiv = sym.get_equivalent_hydrogens()
            self.space_group_name = sym.get_space_group_name()
            self.space_group_number = sym.get_space_group_number()

        for id, atom in enumerate(self.atoms):
            label = c.get_element_label(atom.element)
            labels.append(label)
            c.add_data("atoms", _atom_site_label=
                                    CIF.atom_site_label(label))
            c.add_data("atoms", _atom_site_type_symbol=
                                    CIF.atom_site_type_symbol(atom.element))
            c.add_data("atoms", _atom_site_description=
                                    CIF.atom_site_description(atom.force_field_type))
            c.add_data("atoms", _atom_site_fragment=CIF.atom_site_fragment(atom.sbu_order))
            if self.options.find_symmetric_h:
                if atom.element == "H":
                    symconst = h_equiv[id]
                else:
                    symconst = -1
                c.add_data("atoms", _atom_site_constraints=CIF.atom_site_constraints(symconst))
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

    def mkcell(self, params):
        """Update the cell representation to match the parameters. Currently only 
        builds a 3d cell."""
        self._params = params 
        a_mag, b_mag, c_mag = params[:3]
        alpha, beta, gamma = params[3:] 
        a_vec = np.array([a_mag, 0.0, 0.0])
        b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma), 0.0])
        c_x = c_mag * np.cos(beta)
        c_y = c_mag * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        c_vec = np.array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5])
        self.lattice = np.array([a_vec, b_vec, c_vec])

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

class Symmetry(object):
    def __init__(self, options):
        self.options = options
        assert os.path.isdir(options.symmetry_dir)
        sys.path.append(options.symmetry_dir)
        self.spg = __import__('pyspglib._spglib')._spglib
        #import pyspglib._spglib as spg
        self._symprec = options.symmetry_precision
        self._lattice = None
        self._inv_latt = None
        self._scaled_coords = None
        self._element_symbols = None
        self.dataset = {}

    def add_structure(self, structure):
        self._lattice = structure.cell.lattice.copy()
        self._inv_latt = structure.cell.inverse.copy()
        self._scaled_coords = np.array([atom.in_cell_scaled(self._inv_latt) for
                                        atom in structure.atoms])
        self._angle_tol = -1.0
        self._element_symbols = [atom.element for atom in structure.atoms]
        self._numbers = np.array([ATOMIC_NUMBER.index(i) for i in 
                                    self._element_symbols])

    def refine_cell(self):
        """
        get refined data from symmetry finding
        """
        # Temporary storage of structure info
        _lattice = self._lattice.T.copy()
        _scaled_coords = self._scaled_coords.copy()
        _symprec = self._symprec
        _angle_tol = self._angle_tol
        _numbers = self._numbers.copy()
        
        keys = ('number',
                'international',
                'hall',
                'transformation_matrix',
                'origin_shift',
                'rotations',
                'translations',
                'wyckoffs',
                'equivalent_atoms')
        dataset = {}

        dataset['number'] = 0
        while dataset['number'] == 0:

            # refine cell
            num_atom = len(_scaled_coords)
            ref_lattice = _lattice.copy()
            ref_pos = np.zeros((num_atom * 4, 3), dtype=float)
            ref_pos[:num_atom] = _scaled_coords.copy()
            ref_numbers = np.zeros(num_atom * 4, dtype=int)
            ref_numbers[:num_atom] = _numbers.copy()
            num_atom_bravais = self.spg.refine_cell(ref_lattice,
                                       ref_pos,
                                       ref_numbers,
                                       num_atom,
                                       _symprec,
                                       _angle_tol)
            for key, data in zip(keys, self.spg.dataset(ref_lattice.copy(),
                                    ref_pos[:num_atom_bravais].copy(),
                                ref_numbers[:num_atom_bravais].copy(),
                                            _symprec,
                                            _angle_tol)):
                dataset[key] = data

            _symprec = _symprec * 0.5

        # an error occured with met9, org1, org9 whereby no
        # symmetry info was being printed for some reason.
        # thus a check is done after refining the structure.

        if dataset['number'] == 0:
            warning("WARNING - Bad Symmetry found!")
        else:

            self.dataset['number'] = dataset['number']
            self.dataset['international'] = dataset['international'].strip()
            self.dataset['hall'] = dataset['hall'].strip()
            self.dataset['transformation_matrix'] = np.array(dataset['transformation_matrix'])
            self.dataset['origin_shift'] = np.array(dataset['origin_shift'])
            self.dataset['rotations'] = np.array(dataset['rotations'])
            self.dataset['translations'] = np.array(dataset['translations'])
            letters = "0abcdefghijklmnopqrstuvwxyz"
            try:
                self.dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
            except IndexError:
                print dataset['wyckoffs']
            self.dataset['equivalent_atoms'] = np.array(dataset['equivalent_atoms'])
            self._lattice = ref_lattice.T.copy()
            self._scaled_coords = ref_pos[:num_atom_bravais].copy()
            self._numbers = ref_numbers[:num_atom_bravais].copy()
            self._element_symbols = [ATOMIC_NUMBER[i] for 
                i in ref_numbers[:num_atom_bravais]]

    def get_space_group_name(self):
        return self.dataset["international"] 

    def get_space_group_operations(self):
        return [self.convert_to_string((r, t)) 
                for r, t in zip(self.dataset['rotations'], 
                    self.dataset['translations'])]

    def get_space_group_number(self):
        return self.dataset["number"]

    def get_equiv_atoms(self):
        """Returs a list where each entry represents the index to the
        asymmetric atom. If P1 is assumed, then it just returns a list
        of the range of the atoms."""
        return self.dataset["equivalent_atoms"]

    def get_equivalent_hydrogens(self):
        at_equiv = self.get_equiv_atoms()
        h_equiv = {}
        h_id = list(set([i for id, i in enumerate(at_equiv) 
                    if self._element_symbols[id] == "H"]))
        for id, i in enumerate(self._element_symbols):
            if i == "H":
                h_equiv[id] = h_id.index(at_equiv[id])
        return h_equiv
        #for a in self.get_equiv_atoms():
        #    print a
