# -*- coding: utf-8 -*-
import os

__all__ = ["CSV"]


class CSV:
    def __init__(self, name, _READ=False):
        self._data = {}
        self._headings = []
        if not _READ:
            self.filename = self.get_filename(name)
        else:
            self.filename = name
            self.read()

    def get_filename(self, name):
        if name[-4:] == ".csv":
            base = name[:-4]
        else:
            base = name

        filename = base
        count = 0
        while os.path.isfile(filename + ".csv"):
            count += 1
            filename = base + ".%d" % count
        return filename + ".csv"

    def add_data(self, **kwargs):
        # head_dic = {}
        for key, val in kwargs.items():
            # head_dic.setdefault(key, 0)
            # if key in head_dic.keys():
            #    head_dic[key] += 1
            #    key = "%s.%i"%(key, head_dic[key])
            if key in self._headings:
                self._data.setdefault(key, []).append(val)
            else:
                print("%s not in the headings! Ignoring data!" % (key))

    @property
    def item_count(self):
        lengths = []
        keyss = []
        for key, val in self._data.items():
            keyss.append(key)
            lengths.append(len(val))
        # for i, j in zip(keyss, lengths):
        #    print i,j
        assert all([x == lengths[0] for x in lengths])
        if lengths:
            return lengths[0]
        return 0

    def set_headings(self, *args):
        head_dic = {}
        for head in self._headings:
            name = ".".join(head.split(".")[:-1])
            head_dic.setdefault(name, 0)
            head_dic[name] += 1

        for arg in args:
            head_dic.setdefault(arg, 0)
            if arg in head_dic.keys():
                head_dic[arg] += 1
                arg = "%s.%i" % (arg, head_dic[arg])
            self._headings.append(arg)

    def write(self, filename=None):
        if filename is None:
            f = open(self.filename, "w")
        else:
            f = open(self.get_filename(self.filename), "w")
        # remove the tracking numbers for the final file writing.
        heads = [".".join(i.split(".")[:-1]) for i in self._headings]
        lines = "%s\n" % (",".join(heads))
        for k in range(self.item_count):
            lines += "%s\n" % (
                ",".join(
                    [self.to_str(self._data[i][k]).strip() for i in self._headings]
                )
            )

        f.writelines(lines)
        f.close()

    def to_str(self, val):
        if isinstance(val, str):
            return val
        elif isinstance(val, int):
            return "%i" % val
        elif isinstance(val, float):
            return "%12.6f" % val
        elif isinstance(val, bool):
            return "%d" % val

    def read(self):
        with open(self.filename, "r") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if index == 0:
                    self.set_headings(*[j for j in line.split(",") if j])
                else:
                    self.add_data(
                        **{
                            j: i
                            for j, i in zip(
                                self._headings,
                                [k for k in line.split(",") if not k.startswith("#")],
                            )
                        }
                    )

    def iter_key_vals(self):
        vals = [self._data[j] for j in self._headings]
        for k in zip(*vals):
            yield (zip(self._headings, k))

    def keys(self):
        return self._data.keys()

    def vals(self):
        return self._data.vals()

    def items(self):
        return self._data.items()

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            # print self._data
            print("Error no such key, %s, found in data" % key)
            return None

    def get_row(self, row):
        """Returns row of data ordered by heading sequence"""
        try:
            return [self._data[k][row] for k in self._headings]
        except KeyError:
            return None

    def mofname_dic(self):
        """return a dictionary where the MOFnames are the keys which contain a
        dictionary with the remaining headers specific to those MOFname values,
        only works if MOFname is a header."""
        dic = {}
        try:
            # MOFname.1 becasue we add integers for redundant column headers.
            self._data["MOFname.1"]
        except KeyError:
            print("No MOFname key - returning an empty dictionary")
            return dic
        # remaining headers
        heads = [i for i in self._headings if i != "MOFname.1"]
        for i, name in enumerate(self._data["MOFname.1"]):
            name = self.clean(name)
            dic.setdefault(name, {})
            for j in heads:
                try:
                    dic[name][j] = self._data[j][i]
                except IndexError:
                    print(i, len(self._data[j]))
                    print("No data associated with %s, removing from object" % (name))
                    dic.pop(name)
                    break
        return dic

    def clean(self, name):
        if name.startswith("./run_x"):
            name = name[10:]
        elif name.startswith("run_x"):
            name = name[8:]
        if name.endswith(".cif"):
            name = name[:-4]
        elif name.endswith(".niss"):
            name = name[:-5]
        elif name.endswith(".out-CO2.csv"):
            name = name[:-12]
        elif name.endswith("-CO2.csv"):
            name = name[:-8]
        elif name.endswith(".flog"):
            name = name[:-5]
        elif name.endswith(".out.cif"):
            name = name[:-8]
        elif name.endswith(".tar"):
            name = name[:-4]
        elif name.endswith(".db"):
            name = name[:-3]
        elif name.endswith(".faplog"):
            name = name[:-7]
        elif name.endswith(".db.bak"):
            name = name[:-7]
        elif name.endswith(".csv"):
            name = name[:-4]
        if name.endswith(".out"):
            name = name[:-4]
        return name

    def remove(self, index):
        if isinstance(index, list):
            for k in reversed(sorted(index)):
                for j in self._headings:
                    self._data[j].pop(k)
        else:
            for j in self._headings:
                self._data[j].pop(index)

    @property
    def size(self):
        return len(self._data[self._headings[0]])
