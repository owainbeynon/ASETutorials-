# %% [markdown]
# # Exercise 1: Atomistic Simulations
#
# In this tutorial, we will:
#
# 1. Optimize bulk Pd using the EMT potential.
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
from ase.calculators.emt import EMT
from ase.constraints import StrainFilter
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory
from ase.visualize import view
from ase.io import read
from ase.md.langevin import Langevin
from ase import units
from ase.constraints import FixAtoms

# %% [markdown]
# ## Step 1: Optimize bulk Pd

# %%
# Build bulk Pd (fcc)
element = 'Pd'
atoms = bulk(element, 'fcc', a=3.859)
atoms.calc = EMT()

# Optimize cell with strain filter
sf = StrainFilter(atoms)
opt_bulk = BFGS(sf, trajectory='Pd_bulk_opt.traj', logfile='Pd_bulk_opt.log')
opt_bulk.run(fmax=0.01)

# Optimized lattice constant
optimized_a = atoms.get_cell_lengths_and_angles()[0] * (2 ** 0.5)
print(f"Optimized lattice constant a = {optimized_a:.3f} Å")

view(atoms, viewer='ngl')

# %% [markdown]
# ## Step 2: Build and optimize the Pd(111) surface slab

# %%
# Build Pd(111) surface using optimized lattice constant
supercell_size = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]  # Define the size of the supercell (5x5x5)
supercell= make_supercell(atoms, supercell_size)

view(supercell)
supercell.calc = EMT()

dyn = BFGS(supercell)
dyn.run(fmax=0.01)

# # Fix bottom layers to simulate bulk constraint
# mask = [atom.tag <= 2 for atom in slab]   # True for atoms to fix (bottom layer)
# constraint = FixAtoms(mask=mask)
# slab.set_constraint(constraint)
# %% [markdown]
# ## Step 3: Run a short MD simulation on the slab

# %%
# Initialize velocities corresponding to 300 K
MaxwellBoltzmannDistribution(supercell, temperature_K=300)

# Set up VelocityVerlet dynamics
# Set up Langevin thermostat (NVT ensemble)
dyn = Langevin(supercell,
               timestep=4 * units.fs,      # 5 fs time step
               temperature_K=300,          # Target temperature
               friction=0.02,              # Friction coefficient
               logfile='Pd_md.log')


def printenergy(a=supercell):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(f"Epot = {epot:.3f} eV/atom  Ekin = {ekin:.3f} eV/atom  "
          f"T = {ekin / (1.5 * units.kB):.0f} K  Etot = {epot + ekin:.3f} eV/atom")

# Attach energy printer and trajectory
dyn.attach(printenergy, interval=10)
traj = Trajectory('Pd_slab_md.traj', 'w', supercell)
dyn.attach(traj.write, interval=10)

# Run short MD simulation
print("Starting MD at 300 K...")
printenergy()
dyn.run(100)
print("MD complete!")

# %% [markdown]
# ## Step 4: Optimize isolated H₂O molecule

# %%
h2o = molecule('H2O')
h2o.calc = EMT()

opt_h2o = BFGS(h2o, trajectory='H2O_opt.traj', logfile='H2O_opt.log')
opt_h2o.run(fmax=0.01)

E_h2o = h2o.get_potential_energy()
print(f"Isolated H₂O energy: {E_h2o:.3f} eV")

view(h2o, viewer='ngl')

# %% [markdown]
# ## Step 5: Add H₂O to the Pd(111) surface and relax

# %%

slab = fcc111('Pd', size=(5,5,3), a=optimized_a)

x_center = slab.get_cell()[0,0] / 2
y_center = slab.get_cell()[1,1] / 2


add_adsorbate(slab, h2o, 1.5, position=(x_center, y_center))
slab.center(vacuum=10.0, axis=2)
slab.calc = EMT()

opt_ads = BFGS(slab, trajectory='Pd_H2O_ads.traj', logfile='Pd_H2O_ads.log')
opt_ads.run(fmax=0.01)

E_ads = slab.get_potential_energy()
print(f"Pd(111) + H₂O total energy: {E_ads:.3f} eV")

view(slab, viewer='ngl')

# %% [markdown]
# ## Step 6: Calculate adsorption energy

ads_slab = slab.copy()

ads_slab.calc = EMT()

E_slab = ads_slab.get_potential_energy()
# %%
E_adsorption = E_ads - (E_slab + E_h2o)
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
# using the **VelocityVerlet** integrator.

# %%


# Assign calculator again (just to be sure)
ads_slab.calc = EMT()

# Initialize velocities at 300 K
MaxwellBoltzmannDistribution(ads_slab, temperature_K=300)

# Set up Langevin thermostat (NVT ensemble)
dyn = Langevin(ads_slab,
               timestep=2 * units.fs,      # 5 fs time step
               temperature_K=300,          # Target temperature
               friction=0.02,              # Friction coefficient
               logfile='Pd_H2O_md.log')

# Trajectory output
traj = Trajectory('Pd_H2O_md.traj', 'w', ads_slab)

# Function to print energies every few steps
def print_md_status(a=ads_slab):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    print(f"Epot = {epot:.3f} eV/atom | Ekin = {ekin:.3f} eV/atom | "
          f"T = {temp:.0f} K | Etot = {epot + ekin:.3f} eV/atom")

dyn.attach(print_md_status, interval=10)
dyn.attach(traj.write, interval=1)

# Run 200 steps (~1 ps)
print("Starting MD simulation of H₂O/Pd(111) at 300 K...")
print_md_status()
dyn.run(1000)
print("MD simulation complete!")

# %% [markdown]
# ## Step 9: Visualize the MD trajectory
#
# The trajectory file (`Pd_H2O_md.traj`) can be visualized interactively in ASE or saved for use in tools such as Ovito or VMD.

# %%


md_frames = read('Pd_H2O_md.traj', index=':')
view(md_frames, viewer='ngl')
