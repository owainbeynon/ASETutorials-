# %% [markdown]
# # Exercise 2: Electronic Simulations
#
# In this tutorial, we will:
#
# 1. Optimize bulk Pd using the EMT potential.
# 2. Build a Pd(111) surface slab from the optimized bulk lattice.

from ase.build import fcc111
from ase.visualize import view
from ase.optimize import BFGS
from gpaw import GPAW, PW, restart
from ase.build import bulk, add_adsorbate
from ase.constraints import StrainFilter
import matplotlib.pyplot as plt
from ase.build import molecule

# # %%
# # Build bulk Pd (fcc)
# element = 'Pd'
# atoms = bulk(element, 'fcc', a=3.859)
#
#
# calc = GPAW(mode=PW(300),
#             kpts=(2, 2, 2),
#             xc='PBE',    # exchange-correlation functional
#             txt='pd_slab.out')  # output file
#
#
# atoms.calc = calc
#
# # Optimize cell with strain filter
# sf = StrainFilter(atoms)
# opt_bulk = BFGS(sf, trajectory='Pd_bulk_opt.traj', logfile='Pd_bulk_opt.log')
# opt_bulk.run(fmax=0.01)
# calc.write('pd_slab.gpw')
#
# # Optimized lattice constant
# optimized_a = atoms.get_cell_lengths_and_angles()[0] * (2 ** 0.5)
# print(f"Optimized lattice constant a = {optimized_a:.3f} Å")
#
# view(atoms, viewer='ngl')
#
# calc = GPAW('pd_slab.gpw').fixed_density(
#     nbands=16,
#     symmetry='off',
#     kpts={'path': 'GXWKL', 'npoints': 60},
#     convergence={'bands': 8})
#
#
# bs = calc.band_structure()
# bs.plot(filename='bandstructure.png', show=True, emax=10.0)

#  Slab with CO:
slab = fcc111('Pd', size=(1, 1, 3))

h2o = molecule('H2O')


x_center = slab.get_cell()[0,0] / 2
y_center = slab.get_cell()[1,1] / 2

add_adsorbate(slab, h2o, 1.5, position=(x_center, y_center))
slab.center(axis=2, vacuum=4.0)
slab.calc = GPAW(mode=PW(300),
                 xc='PBE',
                 kpts=(12, 12, 1),
                 convergence={'bands': -10},
                 txt='Pd_slab.txt')
slab.get_potential_energy()
slab.calc.write('Pd_slab.gpw', mode='all')

#  Molecule
h2o = slab[-2:]
h2o.calc = GPAW(mode=PW(300),
                     xc='PBE',
                     kpts=(12, 12, 1),
                     txt='H2O.txt')

molecule.get_potential_energy()
molecule.calc.write('H2O.gpw', mode='all')


# Density of States
plt.subplot(211)
slab, calc = restart('pd_slab.gpw')
e, dos = calc.get_dos(spin=0, npts=2001, width=0.2)
e_f = calc.get_fermi_level()
plt.plot(e - e_f, dos)
plt.axis([-15, 10, None, 4])
plt.ylabel('DOS')

molecule = range(len(slab))[-2:]

plt.subplot(212)
c_mol = GPAW('H2O.gpw')
for n in range(2, 7):
    print('Band', n)
    # PDOS on the band n
    wf_k = [kpt.psit_nG[n] for kpt in c_mol.wfs.kpt_u]
    P_aui = [[kpt.P_ani[a][n] for kpt in c_mol.wfs.kpt_u]
             for a in range(len(molecule))]
    e, dos = calc.get_all_electron_ldos(mol=molecule, spin=0, npts=2001,
                                        width=0.2, wf_k=wf_k, P_aui=P_aui)
    plt.plot(e - e_f, dos, label='Band: ' + str(n))
plt.legend()
plt.axis([-15, 10, None, None])
plt.xlabel('Energy [eV]')
plt.ylabel('All-Electron PDOS')
plt.savefig('pdos.png')
plt.show()