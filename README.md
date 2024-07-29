# Aperiodic Codes

### How to use
* Install module: In the project root folder that contains `setup.py`, run `pip install -e .` ('e' stands for "experimental").
* Upgrade module: In project root folder, run `pip install -U .`
* Core components:
  * [aperiodic_codes/cut_and_project](aperiodic_codes/cut_and_project/): module for constructing codes on 3D cut-and-project (CNP) tilings.
  * [aperiodic_codes/substitution](aperiodic_codes/substitution/): module for constructing self-dual graphical codes on the 2D pinwheel tiling.
  * [scripts](scrips): scripts or jupyter notebooks for examining the code properties.
#### Examples
##### Examine code distance property of classical self-dual graphical codes obtained via nearest-neighbor checking (Laplace operator) on 3D CNP tilings. Verify that anti-Laplacian has linear distance.
1. Install the module.
2. Run [3d_self_dual_distance_property.ipynb](scripts/3d_self_dual_distance_property.ipynb).

##### Check the commutation relation
1. Install the module.
2. generate and save H_z and H_x from [three_dim_from_hgp.py](aperiodic_codes/cut_and_project/three_dim_from_hgp.py). You need to define your own classical codes H1 and H2.
3. use [visualize_commutation.ipynb](scripts/visualize_commutation.ipynb) to examine commutation relations and do any other stuffs you want.
![pinwheel_code](/figures/3d_pinwheel_code.png)
