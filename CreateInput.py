#!/usr/bin/env python
import sys
import os
sys.path.append("/share/apps/openbabel/2.3.1-gg/lib/python2.7/site-packages")
import openbabel as ob
import pybel
from logging import info, debug, warning, error, critical
import numpy
from element_properties import ATOMIC_NUMBER 

def clean(name, ext):
    size = len(ext)+1
    if name[-size:] == "."+ext:
        return name[:-size]
    return name

class InputSBU(object):
    """Contains the necessary information to produce an input for
    Genstruct. This input file is a necessary step in case bonding
    flags or symmetry are incorrect."""
    def __init__(self, filename, ext):
        self.data = {'name':'','index':'', 'metal':'', 'topology':'', 'parent':'',
                'atomic_info':'', 'bond_table':'', 'connectivity':'',
                'connect_flag':'', 'connect_sym':''}
        self.name = clean(filename, ext) 
        self.update(name=self.name)
        self.mol = pybel.readfile(ext, filename).next()
        self._reset_formal_charges()

    def get_index(self):
        ind = self.name[:]
        if "s" == ind[-1:]:
            ind = ind[:-1]
        if "m" == ind[-1:]:
            ind = ind[:-1]
        try:
            ind = int(ind.lstrip("index"))
        except ValueError:
            ind = 0
        self.update(index=str(ind))

    def get_metal(self):
        if "m" in self.name[-2:]:
            self.update(metal="True")
        else:
            self.update(metal="False")

    def special(self):
        """If the mol file ends with an 's', this will interpret
        it as a child SBU, the parent will be the mol name before the 's'"""
        if "s" in self.name[-1:]:
            self.update(parent=self.name[:-1])

    def set_topology(self, top):
        self.update(topology=top)

    def add_data(self, **kwargs):
        self.data.update(kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self.data[key] += val

    def _reset_formal_charges(self):
        """Set all formal charges to zero, this is how special
        information will be passed to oBMol objects."""
        for atom in self.mol:
            atom.OBAtom.SetFormalCharge(0)

    def _remove_atoms(self, *args):
        for obatom in args:
            self.mol.OBMol.DeleteAtom(obatom)

    def get_connect_info(self):
        """Grab all the atoms which are flagged by this program to be
        connectivity points. Namely, Xe, Y, and Rn. Ac series
        elements are replacement Xe atoms for special bonding purposes.
        """
        special, remove = [], []
        connect_index = 0
        for ind, atom in enumerate(self.mol):
            N = atom.atomicnum
            if N == 54 or (N >= 89 and N <= 102):
                connect_index += 1
                con_line = "%4i "%(connect_index)
                X = "%12.4f %8.4f %8.4f"%(atom.coords)
                if (N >= 89 and N <= 102):
                    special.append((connect_index, N%89+1))
                net_vector, bond_vector = "", ""
                for neighbour in ob.OBAtomAtomIter(atom.OBAtom):
                    x = neighbour.GetX() - atom.coords[0]
                    y = neighbour.GetY() - atom.coords[1]
                    z = neighbour.GetZ() - atom.coords[2]
                    if neighbour.GetAtomicNum() == 39:
                        net_atom = neighbour
                        net_vector = "%12.4f %8.4f %8.4f"%(x, y, z)
                        remove.append(net_atom)
                    elif neighbour.GetAtomicNum() == 86:
                        bond_atom = neighbour
                        bond_vector = "%12.4f %8.4f %8.4f"%(x, y, z)
                        remove.append(bond_atom)
                    else:
                        neighbour.SetFormalCharge(connect_index)
                        id = neighbour.GetIdx()

                con_line += "".join([X, bond_vector, net_vector, "\n"])
                self.update(connectivity=con_line)
                remove.append(atom.OBAtom)
        self._remove_atoms(*remove)

        # include special considerations
        for (i, spec) in special:
            if spec == 2:
                bond_partner = 1
            elif spec == 1:
                bond_partner = 2
            else:
                bond_partner = 0
            const_line = '%5i%5i%5i\n'%(i, spec, bond_partner)
            self.update(connect_flag = const_line)

    def get_atom_info(self):
        for atom in self.mol:
            N = atom.OBAtom.GetAtomicNum()
            element = ATOMIC_NUMBER[N]
            coordlines = "%4s  %-6s %8.4f %8.4f %8.4f\n"%(
                    element, self._get_ff_type(atom), atom.coords[0],
                    atom.coords[1], atom.coords[2])
            self.update(atomic_info=coordlines)
            if atom.OBAtom.GetFormalCharge() != 0:
                conn_atom = str(atom.formalcharge) + "C"
                order = "S" # currently set to a single bond.
                tableline = "%4i%4s%4s\n"%(atom.idx-1, conn_atom, order)
                self.update(bond_table=tableline)

    def get_bond_info(self):
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            type = self.return_bondtype(bond)
            line = "%4i%4i%4s\n"%(start_idx-1, end_idx-1, type)
            self.update(bond_table=line)

    def return_bondtype(self, bond):
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if bond.IsSingle():
            return "S"
        elif bond.IsDouble():
            return "D"
        elif bond.IsTriple():
            return "T"
        elif bond.IsAromatic():
            return "A"
        elif start_atom.GetType()[-1] == "R" and end_atom.GetType()[-1] == "R"\
                and start_atom.ExplicitHydrogenCount() == 1 and\
                end_atom.ExplicitHydrogenCount() == 1:
            return "A"
        elif bond.IsAmide():
            return "Am"

    def set_uff(self):
        """Adds UFF atomtyping to the openbabel molecule description"""
        uff = ob.OBForceField_FindForceField('uff')
        uff.Setup(self.mol.OBMol)
        uff.GetAtomTypes(self.mol.OBMol)

    def _get_ff_type(self, pyatom):
       return pyatom.OBAtom.GetData("FFAtomType").GetValue()

    def __str__(self):
        line = "[%(name)s]\nindex = %(index)s\nmetal = %(metal)s\n"%(self.data)
        line += "topology = %(topology)s\n"%(self.data)
        if self.data['parent']:
            line += "parent = %(parent)s\n"%(self.data)
        line += "atoms = \n%(atomic_info)stable = \n"%(self.data)
        line += "%(bond_table)sconnectivity = \n%(connectivity)s"%(self.data)
        if self.data['connect_flag']:
            line += "connect_flag = \n%(connect_flag)s"%(self.data)
        if self.data['connect_sym']:
            line += "connect_sym = \n%(connect_sym)s"%(self.data)
        return line

class SBUFileRead(object):

    def __init__(self, options):
        self.options = options
        self.sbus = []

    def read_sbu_files(self):
        files = self.options.sbu_files if self.options.sbu_files else os.listdir('.')
        files = [file for file in files if 
                 '.'+self.options.file_extension == file[-4:]]
        for f in files:
            info("Reading: %s"%(f))
            s = InputSBU(os.path.basename(f), self.options.file_extension)
            s.get_index()
            s.get_metal()
            s.special()
            if self.options.topologies:
                s.set_topology(self.options.topologies[0])
            else:
                s.set_topology("None")
            s.set_uff()
            s.get_connect_info()
            s.get_atom_info()
            s.get_bond_info()

            self.sbus.append(s)
    def sort_sbus(self):
        """Put metals first, then organics in order of their indices"""
        metals, organics = [], []
        for sbu in self.sbus:
            sbu_ind = int(sbu.data['index'])
            if sbu.data['metal'] == 'True':
                metals.append((sbu_ind, sbu))
            else:
                organics.append((sbu_ind, sbu))
        
        self.sbus = [i[1] for i in sorted(metals)] +\
                [i[1] for i in sorted(organics)]

    def write_file(self):
        filename = os.path.join(self.options.job_dir,self.options.jobname)+ ".out"
        info("writing SBU file to %s"%(filename))
        f = open(filename, "w")
        for sbu in self.sbus:
            f.writelines(str(sbu))
        f.close()

