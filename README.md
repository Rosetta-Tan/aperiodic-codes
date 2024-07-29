# Aperiodic Codes

### How to use
* Installation: `pip install -e setup.py`.
* Core components:
  * [aperiodic-codes/cut_and_project](aperiodic-codes/cut_and_project/): module for constructing codes on 3D cut-and-project (CNP) tilings.
  * [aperiodic-codes/substitution](src/substitution/): module for constructing self-dual graphical codes on the 2D pinwheel tiling.
  * [scripts](scrips): scripts or jupyter notebooks for examining the code properties.
#### Example: examine commutation of 3D CNP X and Z codes from HGP
1. Install the core module.
2. generate and save H_z and H_x from [src/cut_and_project/three_dim_from_hgp.py](src/cut_and_project/three_dim_from_hgp.py). You need to define your own classical codes H1 and H2.
3. use [scripts/visualize_commutation.ipynb](scripts/visualize_commutation.ipynb) to examine commutation relations.
![pinwheel_code](/figures/3d_pinwheel_code.png)
