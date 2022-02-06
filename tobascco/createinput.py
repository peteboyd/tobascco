# -*- coding: utf-8 -*-
import os
from logging import info
from sys import version_info

from openbabel import openbabel as ob

from .element_properties import ATOMIC_NUMBER

__all__ = ["InputSBU"]


def clean(name, ext):
    size = len(ext) + 1
    if name[-size:] == "." + ext:
        return name[:-size]
    return name


class InputSBU(object):
    """Contains the necessary information to produce an input for
    Genstruct. This input file is a necessary step in case bonding
    flags or symmetry are incorrect."""

    def __init__(self, filename, ext):
        self.data = {
            "name": "",
            "index": "",
            "metal": "",
            "topology": "",
            "parent": "",
            "atomic_info": "",
            "bond_table": "",
            "connectivity": "",
            "connect_flag": "",
            "connect_sym": "",
        }
        name = os.path.split(filename)[-1]
        self.name = clean(name, ext)
        self.update(name=self.name)
        # may be a source of error.. untested
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats(ext, "pdb")
        self.mol = ob.OBMol()
        if version_info.major >= 3:
            # self.mol = next(pybel.readfile(ext, filename))
            obConversion.ReadFile(self.mol, filename)
        else:
            obConversion.ReadFile(self.mol, filename)
            # self.mol = pybel.readfile(ext, filename).next()
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
        for atom in ob.OBMolAtomIter(self.mol):
            atom.SetFormalCharge(0)

    def _remove_atoms(self, *args):
        for obatom in args:
            self.mol.DeleteAtom(obatom)

    def get_connect_info(self):
        """Grab all the atoms which are flagged by this program to be
        connectivity points. Namely, Xe, Y, and Rn. Ac series
        elements are replacement Xe atoms for special bonding purposes.
        """
        special, remove = [], []
        connect_index = 0
        for ind, atom in enumerate(ob.OBMolAtomIter(self.mol)):
            N = atom.GetAtomicNum()
            if N == 54 or (N >= 89 and N <= 102):
                connect_index += 1
                con_line = "%4i " % (connect_index)
                X = "%12.4f %8.4f %8.4f" % (atom.GetX(), atom.GetY(), atom.GetZ())
                if N >= 89 and N <= 102:
                    special.append((connect_index, N % 89 + 1))
                net_vector, bond_vector = "", ""
                for neighbour in ob.OBAtomAtomIter(atom):
                    x = neighbour.GetX() - atom.GetX()
                    y = neighbour.GetY() - atom.GetY()
                    z = neighbour.GetZ() - atom.GetZ()
                    if neighbour.GetAtomicNum() == 39:
                        net_atom = neighbour
                        net_vector = "%12.4f %8.4f %8.4f" % (x, y, z)
                        remove.append(net_atom)
                    elif neighbour.GetAtomicNum() == 86:
                        bond_atom = neighbour
                        bond_vector = "%12.4f %8.4f %8.4f" % (x, y, z)
                        remove.append(bond_atom)
                    else:
                        # TEMP if Rn does not exist
                        bond_vector = "%12.4f %8.4f %8.4f" % (-x, -y, -z)
                        neighbour.SetFormalCharge(connect_index)
                        id = neighbour.GetIdx()
                con_line += "".join([X, bond_vector, net_vector, "\n"])
                self.update(connectivity=con_line)
                remove.append(atom)

        self._remove_atoms(*remove)

        # include special considerations
        for (i, spec) in special:
            if spec == 2:
                bond_partner = 1
            elif spec == 1:
                bond_partner = 2
            else:
                bond_partner = 0
            const_line = "%5i%5i%5i\n" % (i, spec, bond_partner)
            self.update(connect_flag=const_line)

    def get_atom_info(self):
        for atom in ob.OBMolAtomIter(self.mol):
            N = atom.GetAtomicNum()
            element = ATOMIC_NUMBER[N]
            coordlines = "%4s  %-6s %8.4f %8.4f %8.4f\n" % (
                element,
                self._get_ff_type(atom),
                atom.GetX(),
                atom.GetY(),
                atom.GetZ(),
            )
            self.update(atomic_info=coordlines)
            if atom.GetFormalCharge() != 0:
                conn_atom = str(atom.GetFormalCharge()) + "C"
                order = "S"  # currently set to a single bond.
                tableline = "%4i%4s%4s\n" % (atom.GetIdx() - 1, conn_atom, order)
                self.update(bond_table=tableline)

    def get_bond_info(self):
        for bond in ob.OBMolBondIter(self.mol):
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            type = self.return_bondtype(bond)
            line = "%4i%4i%4s\n" % (start_idx - 1, end_idx - 1, type)
            self.update(bond_table=line)

    def return_bondtype(self, bond):
        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        order = bond.GetBondOrder()
        # if bond.IsSingle():
        if order == 1:
            return "S"
        # elif bond.IsDouble():
        if order == 2:
            return "D"
        # elif bond.IsTriple():
        if order == 3:
            return "T"
        elif bond.IsAromatic():
            return "A"
        elif (
            start_atom.GetType()[-1] == "R"
            and end_atom.GetType()[-1] == "R"
            and start_atom.ExplicitHydrogenCount() == 1
            and end_atom.ExplicitHydrogenCount() == 1
        ):
            return "A"
        elif bond.IsAmide():
            return "Am"

    def set_uff(self):
        """Adds UFF atomtyping to the openbabel molecule description"""
        uff = ob.OBForceField_FindForceField("uff")
        uff.Setup(self.mol)
        uff.GetAtomTypes(self.mol)

    def _get_ff_type(self, pyatom):
        return pyatom.GetData("FFAtomType").GetValue()

    def __str__(self):
        line = "[%(name)s]\nindex = %(index)s\nmetal = %(metal)s\n" % (self.data)
        line += "topology = %(topology)s\n" % (self.data)
        if self.data["parent"]:
            line += "parent = %(parent)s\n" % (self.data)
        line += "atoms = \n%(atomic_info)stable = \n" % (self.data)
        line += "%(bond_table)sconnectivity = \n%(connectivity)s" % (self.data)
        if self.data["connect_flag"]:
            line += "connect_flag = \n%(connect_flag)s" % (self.data)
        if self.data["connect_sym"]:
            line += "connect_sym = \n%(connect_sym)s" % (self.data)
        return line


class SBUFileRead(object):
    def __init__(self, options):
        self.options = options
        self.sbus = []

    def read_sbu_files(self):
        files = []
        ext_len = len(self.options.file_extension.strip())
        if self.options.sbu_files:
            for sbuf in self.options.sbu_files:
                if sbuf[-ext_len:] == "." + self.options.file_extension:
                    files.append(sbuf)
                else:
                    if os.path.isdir(os.path.abspath(sbuf)):
                        for j in os.listdir(os.path.abspath(sbuf)):
                            if j[-ext_len:] == "." + self.options.file_extension:
                                files.append(os.path.join(os.path.abspath(sbuf), j))

        else:
            files = [
                j
                for j in os.listdir(os.getcwd())
                if j.endswith(self.options.file_extension)
            ]
        for f in files:
            info("Reading: %s" % (os.path.basename(f)))
            s = InputSBU(f, self.options.file_extension)
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
            sbu_ind = int(sbu.data["index"])
            if sbu.data["metal"] == "True":
                metals.append((sbu_ind, sbu))
            else:
                organics.append((sbu_ind, sbu))

        self.sbus = [i[1] for i in sorted(metals)] + [i[1] for i in sorted(organics)]

    def write_file(self):
        filename = os.path.join(self.options.job_dir, self.options.jobname) + ".out"
        info("writing SBU file to %s" % (filename))
        with open(filename, "w") as handle:
            for sbu in self.sbus:
                handle.writelines(str(sbu))
