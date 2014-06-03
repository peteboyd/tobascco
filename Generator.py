#!/usr/bin/env python
import itertools
from random import choice
from SecondaryBuildingUnit import SBU_list
import sys


class Generate(object):
    """Takes as input a sequence of sbus, and returns
    build orders to make structures.
    
    """
    
    def __init__(self, options, sbu_list):
        self.options = options
        self.sbus = SBU_list(sbu_list)
        
    def generate_sbu_combinations(self, incidence=None, N=None):
        if N is None:
            N = self.options.metal_sbu_per_structure + self.options.organic_sbu_per_structure
        for i in itertools.combinations_with_replacement(self.sbus.list, N):
            if self._valid_sbu_combination(incidence, i):
                yield tuple(i)

    def combinations_from_options(self):
        """Just return the tuples in turn."""
        combs = []
        for combo in self.options.sbu_combinations:
            # first one has to be a metal.
            met = [self.sbus.get(combo[0], _METAL=True)]
            combs.append(tuple(met + [self.sbus.get(i) for i in combo[1:]]))
        return combs
    
    def _valid_sbu_combination(self, incidence, sbu_set):
        """Currently only checks if there is the correct number of metal 
        SBUs in the combination."""
        if incidence is None:
            return len([i for i in sbu_set if i.is_metal]) == \
                        self.options.metal_sbu_per_structure
        else:
            if set(sorted([i.degree for i in sbu_set])) == set(sorted(incidence)):
                return len([i for i in sbu_set if i.is_metal]) == \
                                self.options.metal_sbu_per_structure
            else:
                return False

    def yield_linear_org_sbu(self, combo):
        for i in self.sbus.list:
            if i.linear and not i.is_metal: 
                ret = list(combo) + [i]
                yield tuple(ret)
   
    @property
    def linear_sbus_exist(self):
        try:
            return self._linear_exist
        except AttributeError:
            self._linear_exist = False
            for i in self.sbus.list:
                if i.linear:
                    self._linear_exist = True
                    break
            return self._linear_exist

    def _valid_bond_pair(self, set):
        """Determine if the two SBUs can be bonded.  Currently set to
        flag true if the two sbus contain matching bond flags, otherwise
        if they are a (metal|organic) pair
        """
        (sbu1, cp1), (sbu2, cp2) = set
        if all([i is None for i in [cp1.special, cp2.special, cp1.constraint, cp2.constraint]]):
            return sbu1.is_metal != sbu2.is_metal

        return (cp1.special == cp2.constraint) and (cp2.special == cp1.constraint)
        
