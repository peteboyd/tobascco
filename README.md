# tobascco

## Installation

You need to first install openbabel and nlopt (we're working on a conda recipe to skip this step):

```
conda install -c conda-forge openbabel nlopt
```

Then you can install the package

```
pip install git+https://github.com/peteboyd/tobascco
```

This automatically installs a runscript that is appropriate for most use cases

## Usage

### Assembling MOFs

### SBU databases

A key part of the code are the SBU databases (metal nodes, organic linkers) some defaults are shipped with the package. Those databases are currently file based, i.e., plain text files that need to be parsed for every run of the code and to which new SBUs need to be appended.

#### Extending the SBU database

New entries to the SBU database can be added using the code in `createinput.py` module (or job type `create_sbu_input_files=True` in an input file).

## Reference

If you use this code, please cite [Boyd, P. G.; K. Woo, T. A Generalized Method for Constructing Hypothetical Nanoporous Materials of Any Net Topology from Graph Theory. CrystEngComm 2016, 18 (21), 3777–3792.](https://pubs.rsc.org/--/content/articlelanding/2016/ce/c6ce00407e)
