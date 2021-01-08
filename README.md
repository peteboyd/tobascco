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

A key part of the code are the SBU databases (metal nodes, organic linkers)

#### Extending the SBU database

New entries to the SBU database can be added using the code in `createinput.py` module (or job type `create_sbu_input_files=True` in an input file

## Reference

If you use this code, please cite [Boyd, P. G.; K. Woo, T. A Generalized Method for Constructing Hypothetical Nanoporous Materials of Any Net Topology from Graph Theory. CrystEngComm 2016, 18 (21), 3777–3792.](https://doi.org/10.1039/C6CE00407E.)
