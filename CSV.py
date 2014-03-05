#!/usr/bin/env python
import os

class CSV(object):
    
    def __init__(self, name):
        self.filename = self.get_filename(name)
        self._data = {}
        self._headings = []
        
    def get_filename(self, name):
        if name[-4:] == ".csv":
            base = name[:-4]
        else:
            base = name

        filename = base
        count = 0
        while os.path.isfile(filename + '.csv'):
            count += 1
            filename = base + ".%d"%count
        return filename + '.csv'

    def add_data(self, **kwargs):
        for key, val in kwargs.items():
            assert key in self._headings, "%s not in the headings: "%(key) + \
                                            ", ".join(self._headings)
            self._data.setdefault(key, []).append(val)
        
    @property
    def item_count(self):
        lengths = []
        for key, val in self._data.items():
            lengths.append(len(val))
        assert all([x == lengths[0] for x in lengths])
        return lengths[0]
        
    def set_headings(self, *args):
        [self._headings.append(arg) for arg in args]
        
    def write(self):
        f = open(self.filename, 'w')
        lines = "%s\n"%(','.join(self._headings))
        
        for k in range(self.item_count):    
            lines += "%s\n"%(','.join([self.to_str(self._data[i][k]).strip() for i in self._headings]))
            
        f.writelines(lines)
        f.close()
        
    def to_str(self, val):
        if isinstance(val, str):
            return val
        elif isinstance(val, int):
            return "%i"%val
        elif isinstance(val, float):
            return "%12.6f"%val
        elif isinstance(val, bool):
            return "%d"%val
