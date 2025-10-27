# %% [markdown]
# # Exercise 1: Machine Learning Interatomic Potentials
#
# In this tutorial, we will:
#
# 1. Optimize bulk Pd using the MACE MP0 model
# 2. Build a Pd(111) surface slab from the optimized bulk lattice.
# 3. Run a short molecular dynamics (MD) simulation on the slab.
# 4. Add a water molecule as an adsorbate and relax the system.
# 5. Calculate the adsorption energy:
#
# \[
# E_\text{ads} = E_\text{slab+H₂O} - (E_\text{slab} + E_\text{H₂O})
# \]
#
# A negative value indicates favorable adsorption (exothermic).

# %%
from ase import units
from ase.build import bulk, make_supercell, fcc111, molecule, add_adsorbate
from mace.calculators import mace_mp
from ase.constraints import StrainFilter
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase.visualize import view
from ase.io import read
from ase.md.langevin import Langevin
from ase import units
from ase.constraints import FixAtoms

from ASETutorials.scripts.exercise1 import ads_slab

# %% [markdown]
# ## Step 1: Optimize bulk Pd

# %%
# Build bulk Pd (fcc)
element = 'Pd'
atoms = bulk(element, 'fcc', a=3.859)

macemp = mace_mp(dispersion=True, float=64)

atoms.calc = macemp

# Optimize cell with strain filter
sf = StrainFilter(atoms)
opt_bulk = BFGS(sf, trajectory='Pd_bulk_opt.traj', logfile='Pd_bulk_opt.log')
opt_bulk.run(fmax=0.01)

# Optimized lattice constant
optimized_a = atoms.get_cell_lengths_and_angles()[0] * (2 ** 0.5)
print(f"Optimized lattice constant a = {optimized_a:.3f} Å")

view(atoms, viewer='ngl')

# # Fix bottom layers to simulate bulk constraint
# mask = [atom.tag <= 2 for atom in slab]   # True for atoms to fix (bottom layer)
# constraint = FixAtoms(mask=mask)
# slab.set_constraint(constraint)
# %% [markdown]
# ## Step 3: Run a short MD simulation on the slab

# %% [markdown]
# ## Step 4: Optimize isolated H₂O molecule

# %%
h2o = molecule('H2O')
h2o.calc = macemp

opt_h2o = BFGS(h2o, trajectory='H2O_opt.traj', logfile='H2O_opt.log')
opt_h2o.run(fmax=0.01)

E_h2o = h2o.get_potential_energy()
print(f"Isolated H₂O energy: {E_h2o:.3f} eV")

view(h2o, viewer='ngl')

# %% [markdown]
# ## Step 5: Add H₂O to the Pd(111) surface and relax

# %%

slab = fcc111('Pd', size=(5,5,3), a=optimized_a)

slab.calc = macemp

E_slab = slab.get_potential_energy()

x_center = slab.get_cell()[0,0] / 2
y_center = slab.get_cell()[1,1] / 2

add_adsorbate(slab, h2o, 1.5, position=(x_center, y_center))
slab.center(vacuum=10.0, axis=2)
slab.calc = macemp

opt_ads = BFGS(slab, trajectory='Pd_H2O_ads.traj', logfile='Pd_H2O_ads.log')
opt_ads.run(fmax=0.01)

E_Pd_H2O = slab.get_potential_energy()
print(f"Pd(111) + H₂O total energy: {E_Pd_H2O:.3f} eV")

view(slab, viewer='ngl')

# %% [markdown]
# ## Step 6: Calculate adsorption energy
# %%
E_adsorption = E_slab - (E_Pd_H2O + E_h2o)
print(f"Adsorption energy of H₂O on Pd(111): {E_adsorption:.3f} eV")

# %% [markdown]
# ## Step 7: Summary
#
# | Quantity | Symbol | Energy (eV) |
# |-----------|---------|-------------|
# | Clean slab | E_slab | `{E_slab:.3f}` |
# | Isolated H₂O | E_h2o | `{E_h2o:.3f}` |
# | Adsorbed system | E_ads | `{E_ads:.3f}` |
# | **Adsorption energy** | **E_adsorption** | **`{E_adsorption:.3f}`** |
#
# Negative adsorption energy → exothermic adsorption (favorable binding).

# %% [markdown]
# ## Step 8: Run a molecular dynamics (MD) simulation of H₂O on Pd(111)
#
# Now that the adsorbed structure has been optimized, we'll perform a short molecular
# dynamics (MD) simulation to study the thermal motion of H₂O on the Pd(111) surface.
#
# We'll initialize the system at 300 K and propagate it for a few hundred femtoseconds

# %%

ads_slab = slab.copy()
# Assign calculator again (just to be sure)
ads_slab.calc = macemp

# Initialize velocities at 300 K
MaxwellBoltzmannDistribution(ads_slab, temperature_K=300)

# Set up Langevin thermostat (NVT ensemble)
dyn = Langevin(ads_slab,
               timestep=1 * units.fs,      # 5 fs time step
               temperature_K=300,          # Target temperature
               friction=0.02,              # Friction coefficient
               logfile='Pd_H2O_md.log',
               trajectory='Pd_H2O_md.traj')

# Trajectory output

dyn.run(1000)
print("MD simulation complete!")

# %% [markdown]
# ## Step 9: Visualize the MD trajectory
#
# The trajectory file (`Pd_H2O_md.traj`) can be visualized interactively in ASE or saved for use in tools such as Ovito or VMD.

# %%


md_frames = read('Pd_H2O_md.traj', index=':')
view(md_frames, viewer='ngl')
