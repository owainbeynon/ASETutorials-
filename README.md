# ASE  Tutorials

Here are a set of tutorials to get started with ASE - a Python library for atomistic simulations. 

ASE Documentations can be found here: https://ase-lib.org. 

To get started launch the tutorials in binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/owainbeynon/ASETutorials-/main?urlpath=lab)

In this repo you'll find three exercises taking you across the materials modelling Scale. 

## Exercise 1: Atomistic Modelling 

In this example, we build a simple metallic system, run a geometry optimisation, vibrational and molecular dynamics simulation using effective mean theory (EMT) - a classical potential. 
Choose from FCC Al, Cu, Au, Ni, Ag, Pt and Pd. You can find the lattice constant (a) for each one in a table below:



| Metal     | Unit Cell | Lattice constant (Å) |
|-----------|-----------|----------------------|
| Aluminum  | FCC       | 4.046                |
| Copper    | FCC       | 3.597                |
| Gold      | FCC       | 4.065                |
| Nickel    | FCC       | 3.499                |
| Silver    | FCC       | 4.079                |
| Platinum  | FCC       | 3.912                |
| Palladium | FCC       | 3.859                |


Some useful reading of water adsorption on metallic surfaces 

1) DFT study of the adsorption and dissociation of water on Ni(111), Ni(110) and Ni(100) surface:  https://doi.org/10.1016/j.susc.2014.04.006 
2) Interaction of H2O with the Platinum Pt (001), (011), and (111) Surfaces: A Density Functional Theory Study with Long-Range Dispersion Correction: https://doi.org/10.1021/acs.jpcc.9b06136
3) Evaluation of the onset voltage of water adsorption on Pt(111) surface using density functional theory/implicit model calculations: https://doi.org/10.1016/j.surfin.2025.105809
4) Adsorption of Water on Cu(100) and Pd(100) at Low Temperatures: Observation of Monomeric Water: https://doi.org/10.1016/S0167-2991(09)61214-3
5) Water adsorption on Pd {100} from first principles: https://doi.org/10.1103/PhysRevB.76.235433

## Exercise 2: Electronic Modelling 

Here we will use GPAW (https://gpaw.readthedocs.io) a Python-based DFT code for looking at the properties of H2O-metal adsorption. 

1) Obtain band structure 
2) Obtain H2O-Pd DOS



## Exercise 3: Machine Learning Interatomic Potentials 

Machine learning interatomic potentials (MLIPs) combine the efficiency of DFT with the speed of classical potentials, effectively 
bypassing the cost-accuracy of materials modelling. 

Here we will use MACE architecture (https://mace-docs.readthedocs.io/en/latest/) to rerun the examples of Exercise 1 with out-of-the-box MACE foundation models MP0,
trained on structures adn energy from the materials project database (https://next-gen.materialsproject.org)


1) how do the energies and values obtain compare to DFT and EMT?



