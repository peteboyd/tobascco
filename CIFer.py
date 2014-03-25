#!/usr/bin/env python
from datetime import date

class CIF(object):

    def __init__(self, name="structure"):
        self.name = name
        self._data = {}
        self._headings = {}
        self._element_labels = {}
        self.non_loops = ["data", "cell", "sym"]
        self.block_order = ["data", "sym", "sym_loop", "cell", "atoms", "bonds"]

    def get_time(self):
        t = date.today()
        return t.strftime("%A %d %B %Y")

    def insert_block_order(self, name, index=None):
        """Adds a block to the cif file in a specified order"""
        if index is None:
            index = len(self.block_order)
        self.block_order = self.block_order[:index] + [name] + \
                            self.block_order[index:]

    def add_data(self, block, **kwargs):
        self._headings.setdefault(block, [])
        for key, val in kwargs.items():
            try:
                self._data[key].append(val)
            except KeyError:
                self._headings[block].append(key)
                if block in self.non_loops:
                    self._data[key] = val
                else:
                    self._data[key] = [val]

    def get_element_label(self, el):
        self._element_labels.setdefault(el, 0)
        self._element_labels[el] += 1
        return el + str(self._element_labels[el])

    def __str__(self):
        line = ""
        for block in self.block_order:
            heads = self._headings[block]
            if block in self.non_loops: 
                vals = zip([CIF.label(i) for i in heads], [self._data[i] for i in heads])
            else:
                line += "loop_\n"+"\n".join([CIF.label(i) for i in heads])+"\n"
                vals = zip(*[self._data[i] for i in heads])
            for ll in vals:
                line += "".join(ll) + "\n"
        return line

    # terrible idea for formatting.. but oh well :)
    @staticmethod
    def atom_site_fract_x(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_y(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_z(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_label(x):
        return "%-7s "%(x)
    @staticmethod
    def atom_site_type_symbol(x):
        return "%-6s "%(x)
    @staticmethod
    def atom_site_description(x):
        return "%-5s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_1(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_2(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_distance(x):
        return "%7.3f "%(x)
    @staticmethod
    def geom_bond_site_symmetry_2(x):
        return "%-5s "%(x)
    @staticmethod
    def ccdc_geom_bond_type(x):
        return "%5s "%(x)
    @staticmethod
    def cell_length_a(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_b(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_c(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_alpha(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_beta(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_gamma(x):
        return "%-7.4f "%(x)
    @staticmethod
    def atom_site_fragment(x):
        return "%-4i "%(x)
    @staticmethod
    def atom_site_constraints(x):
        return "%-4i "%(x)

    @staticmethod
    def label(x):
        """special cases"""
        if x == "data_":
            return x
        elif x == "_symmetry_space_group_name_H_M":
            # replace H_M with H-M. 
            x = x[:28] + "-" + x[29:]
        return "%-34s"%(x)
