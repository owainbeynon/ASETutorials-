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
from ase.build import bulk, make_supercell, fcc111, molecule, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import StrainFilter, FixAtoms
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io import Trajectory, read
from ase.visualize import view
from ase import units

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
optimized_a = atoms.get_cell_lengths_and_ax3des()[0]
print(f"Optimized lattice constant a = {optimized_a:.3f} Å")

view(atoms, viewer='x3d')

# ## Step 2: Build and optimize the Pd(111) surface slab

# %%
# Build Pd(111) surface using optimized lattice constant
slab = fcc111('Pd', size=(5,5,3), a=optimized_a)
slab.calc = EMT()

# Optimize slab
opt_slab = BFGS(slab, trajectory='Pd_slab_opt.traj', logfile='Pd_slab_opt.log')
opt_slab.run(fmax=0.01)

E_slab = slab.get_potential_energy()
print(f"Clean Pd(111) slab energy: {E_slab:.3f} eV")

view(slab, viewer='x3d')


# ## Step 3: Short MD simulation of the clean slab

# %%
# Initialize velocities at 300 K
MaxwellBoltzmannDistribution(slab, temperature_K=300)

# Set up Langevin dynamics
dyn = Langevin(
    slab,
    timestep=4 * units.fs,
    temperature_K=300,
    friction=0.02,
    logfile='Pd_slab_md.log'
)

# Function to print energies per atom
def print_energy(a=slab):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    print(f"Epot = {epot:.3f} eV/atom | Ekin = {ekin:.3f} eV/atom | T = {temp:.0f} K | Etot = {epot + ekin:.3f} eV/atom")

# Attach energy printer and trajectory
dyn.attach(print_energy, interval=10)
traj = Trajectory('Pd_slab_md.traj', 'w', slab)
dyn.attach(traj.write, interval=10)

print("Starting MD of clean Pd(111) slab at 300 K...")
print_energy()
dyn.run(100)
print("MD complete!")

# ## Step 4: Optimize isolated H₂O molecule

# %%
h2o = molecule('H2O')
h2o.calc = EMT()

opt_h2o = BFGS(h2o, trajectory='H2O_opt.traj', logfile='H2O_opt.log')
opt_h2o.run(fmax=0.01)

E_h2o = h2o.get_potential_energy()
print(f"Isolated H₂O energy: {E_h2o:.3f} eV")

view(h2o, viewer='x3d')

# ## Step 5: Add H₂O to the Pd(111) surface and relax

# Center H2O above slab
x_center = slab.get_cell()[0,0] / 2
y_center = slab.get_cell()[1,1] / 2
add_adsorbate(slab, h2o, 1.5, position=(x_center, y_center))
slab.center(vacuum=10.0, axis=2)
slab.calc = EMT()

# Optimize adsorbed system
opt_ads = BFGS(slab, trajectory='Pd_H2O_ads.traj', logfile='Pd_H2O_ads.log')
opt_ads.run(fmax=0.01)

E_Pd_H2O = slab.get_potential_energy()
print(f"Pd(111) + H₂O total energy: {E_Pd_H2O:.3f} eV")

view(slab, viewer='x3d')

# ## Step 6: Calculate adsorption energy

E_ads = E_Pd_H2O - (E_slab + E_h2o)
print(f"Adsorption energy of H₂O on Pd(111): {E_ads:.3f} eV")

# ## Step 7: Summary
#
# Print the table
print(f"{'Quantity':<20} | {'Symbol':<10} | {'Energy (eV)':<12}")
print("-" * 50)
print(f"{'Clean slab':<20} | {'E_slab':<10} | {E_slab:>12.3f}")
print(f"{'Isolated H2O':<20} | {'E_h2o':<10} | {E_h2o:>12.3f}")
print(f"{'Adsorbed system':<20} | {'E_ads':<10} | {E_Pd_H2O:>12.3f}")
print(f"{'Adsorption energy':<20} | {'E_adsorption':<10} | {E_ads:>12.3f}")
print("\nNegative adsorption energy → exothermic adsorption (favorable binding).")


# ## Step 8: MD simulation of H₂O on Pd(111)

ads_slab = slab.copy()
ads_slab.calc = EMT()

# Initialize velocities
MaxwellBoltzmannDistribution(ads_slab, temperature_K=300)

# Langevin dynamics
dyn = Langevin(
    ads_slab,
    timestep=2 * units.fs,
    temperature_K=300,
    friction=0.02,
    logfile='Pd_H2O_md.log'
)

traj = Trajectory('Pd_H2O_md.traj', 'w', ads_slab)

def print_md_status(a=ads_slab):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    print(f"Epot = {epot:.3f} eV/atom | Ekin = {ekin:.3f} eV/atom | T = {temp:.0f} K | Etot = {epot + ekin:.3f} eV/atom")

dyn.attach(print_md_status, interval=10)
dyn.attach(traj.write, interval=1)

print("Starting MD simulation of H₂O/Pd(111) at 300 K...")
print_md_status()
dyn.run(1000)
print("MD simulation complete!")

# ## Step 9: Visualize the MD trajectory

# %%
md_frames = read('Pd_H2O_md.traj', index=':')
view(md_frames, viewer='x3d')
