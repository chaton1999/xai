#!/usr/bin/env python3
import argparse
import numpy as np
from ase.io import Trajectory
from NNCalculator.NNCalculator import NNCalculator

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input", type=str, help="Input trajectory file", required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge", type=int, help="Total charge", default=0)
args = parser.parse_args()

print("Input file: {}".format(args.input))

# === Load trajectory ===
traj = Trajectory(args.input)
n_frames = len(traj)
atoms = traj[0]

# === Initialize the calculator ===
calc = NNCalculator(
    checkpoint="ll_models_ccsdt/ma_ccsd-t_tl_dataset_extended_final3_862_number9",
    atoms=atoms,
    charge=args.charge,
    F=128,
    K=64,
    num_blocks=5,
    num_residual_atomic=2,
    num_residual_interaction=3,
    num_residual_output=1,
    sr_cut=10.0,
    use_electrostatic=True,
    use_dispersion=False,
    s6=1.0000,
    s8=2.3550,
    a1=0.5238,
    a2=3.5016
)

# === Process and save per-frame outputs ===
for i, atoms in enumerate(traj):
    atoms.set_calculator(calc)

    # Energy and forces
    potential_energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    charges = atoms.get_charges()
    mu = np.dot(atoms.get_charges(), atoms.get_positions())  # Dipole moment calculation 
    if len(mu) == 2:  # If only two components are calculated, append a zero for the z-component
        mu = np.append(mu, 0.0)

    # Neural network internal quantities
    act_frame = np.array(calc.representation)
    atomic_energies = np.array(calc.atomic_energies)
    electrostatic_energies = np.array(calc.electrostatic_energies)

    # Compute decomposition total and inferred dispersion
    E_decomp_total = np.sum(atomic_energies) + np.sum(electrostatic_energies)
    dispersion_energy = potential_energy - E_decomp_total
    
    dispersion_per_atom = np.array(calc.dispersion_energies)  # eV per atom
    dispersion_total = dispersion_per_atom.sum()
    # np.isclose(potential_energy, atomic_energies.sum() + electrostatic_energies.sum() + dispersion_total, atol=1e-5)
    
    print(dispersion_total,dispersion_energy)
    # Save everything
    npz_file = "test_Edisp_{}.xyz.npz".format(i + 1)
    np.savez(
        npz_file,
        act=act_frame,
        F=forces,
        Q=charges,
        D= mu, 
        E=np.array([potential_energy]),
        E_atom=atomic_energies,
        E_elec=electrostatic_energies,
        E_disp=dispersion_per_atom,
        E_dtot=np.array([dispersion_energy]),
        E_dsum=dispersion_total,
        E_sum=np.array([E_decomp_total + dispersion_energy])
    )

    print("Frame {:3d}: saved to {}".format(i + 1, npz_file))

print("✅ All frames processed and saved.")

