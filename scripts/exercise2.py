# # Exercise 2: Electronic Simulations
#
# In this tutorial, we will:
# 1. Optimize bulk Pd using the PBE XC Functional
# 2. Build a Pd(111) surface slab from the optimized bulk lattice and add a water molecule
# 3. Calculate adsorption energy (E_ads) and density of states (DOS)

# Imports
from ase.build import fcc111, bulk, add_adsorbate, molecule
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import StrainFilter
from gpaw import GPAW, PW
import matplotlib.pyplot as plt
from ase.io import write

# Step 1: Build bulk Pd (fcc) and optimize lattice constant
element = 'Pd'
atoms = bulk(element, 'fcc', a=3.859)

calc = GPAW(mode=PW(300),
            kpts=(3, 2, 2),
            xc='PBE',
            txt='pd_bulk.txt')

atoms.calc = calc

sf = StrainFilter(atoms)
opt_bulk = BFGS(sf, trajectory='Pd_bulk_opt.traj', logfile='Pd_bulk_opt.log')
opt_bulk.run(fmax=0.01)
calc.write('pd_bulk.gpw')

optimized_a = atoms.get_cell_lengths_and_angles()[0] * (2 ** 0.5)
print(f"Optimized lattice constant a = {optimized_a:.3f} Å")

view(atoms, viewer='x3d')

# Step 2: Build Pd(111) slab
slab = fcc111('Pd', size=(3, 3, 2), vacuum=10.0, a=optimized_a)
write('Pd.xyz', slab)

calc = GPAW(mode=PW(300),
            xc='PBE',
            kpts=(2, 2, 1),
            txt='water_slab.txt')

slab.calc = calc
E_slab = slab.get_potential_energy()
print(f"Slab energy: {E_slab:.6f} eV")

# Step 3: Calculate isolated H2O molecule energy
h2o = molecule('H2O')
h2o.center(vacuum=4.0)
h2o.pbc = False

h2o.calc = GPAW(mode=PW(300),
                xc='PBE',
                kpts=(1, 1, 1),
                txt='H2O.txt')

E_h2o = h2o.get_potential_energy()
h2o.calc.write('H2O.gpw', mode='all')
print(f"H2O molecule energy: {E_h2o:.6f} eV")

# Step 4: Add H2O to Pd(111) slab and optimize
x_center = slab.get_cell()[0, 0] / 2
y_center = slab.get_cell()[1, 1] / 2

add_adsorbate(slab, h2o, height=1.5, position=(x_center, y_center))
slab.center(axis=2, vacuum=10.0)
write('Pd_h2o_initial.xyz', slab)

calc = GPAW(mode=PW(300),
            xc='PBE',
            kpts=(2, 2, 1),
            txt='Pd_H2O.txt')

slab.calc = calc

opt = BFGS(slab, trajectory='Pd_H2O_opt.traj', logfile='Pd_H2O_opt.log')
opt.run(fmax=0.01)

E_Pd_H2O = slab.get_potential_energy()
slab.calc.write('Pd_slab.gpw', mode='all')
write('Pd_h2o_optimized.xyz', slab)

print(f"Pd–H2O system energy: {E_Pd_H2O:.6f} eV")

# Step 5: Compute adsorption energy
E_ads = E_Pd_H2O - (E_slab + E_h2o)
print(f"Adsorption energy of H₂O on Pd(111): {E_ads:.3f} eV")

# Step 6: Summary table
print(f"{'System':<20} | {'Symbol':<10} | {'Energy (eV)':<12}")
print("-" * 50)
print(f"{'Clean slab':<20} | {'E_slab':<10} | {E_slab:>12.3f}")
print(f"{'Isolated H2O':<20} | {'E_h2o':<10} | {E_h2o:>12.3f}")
print(f"{'Adsorbed system':<20} | {'E_Pd_H2O':<10} | {E_Pd_H2O:>12.3f}")
print(f"{'Adsorption energy':<20} | {'E_ads':<10} | {E_ads:>12.3f}")
print("\nNegative adsorption energy → exothermic adsorption (favorable binding).")
