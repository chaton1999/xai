#!/usr/bin/env python3
import argparse
import numpy as np
from ase.io import Trajectory
from NNCalculator_energy_module.NNCalculator import NNCalculator


def _as_scalar(x):
    """Return a Python float from a numpy scalar/array; sum if it's a vector."""
    x = np.asarray(x)
    if x.size == 1:
        return float(x.reshape(()))  # or x.item()
    return float(np.sum(x))

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
atoms0 = traj[0]

# === Initialize the calculator ===
calc = NNCalculator(
    checkpoint="model.ckpt-1000000",
    atoms=atoms0,
    charge=args.charge,
    F=128,
    K=64,
    num_blocks=5,
    num_residual_atomic=2,
    num_residual_interaction=3,
    num_residual_output=1,
    sr_cut=10.0,
    # If you also want a physical nonbonded cutoff for Coulomb/dispersion, add: lr_cut=10.0,
    use_electrostatic=True,
    use_dispersion=False,   # keep False if you trained/evaluate without D3
    s6=1.0000,
    s8=2.3550,
    a1=0.5238,
    a2=3.5016
)

# === Process and save per-frame outputs ===
for i, atoms in enumerate(traj):
    atoms.set_calculator(calc)

    # Energies, forces, charges
    E_tot = atoms.get_potential_energy()
    F = atoms.get_forces()
    Q = atoms.get_charges()
    R = atoms.get_positions()
    Z = atoms.get_atomic_numbers()

    # Dipole (from predicted charges); pad to 3 components if needed
    mu = np.dot(Q, R)
    mu = np.asarray(mu)
    if mu.shape == (2,):
        mu = np.append(mu, 0.0)

    # --- Internal quantities from the calculator ---
    act_frame = np.array(calc.representation)         # features per block
    E_atom = np.array(calc.atomic_energies)           # per-atom NN (scaled+shift)
    E_elec_atom = np.array(calc.electrostatic_energies)

    # Dispersion (0 if not exposed / disabled)
    try:
        E_disp_atom = np.array(calc.dispersion_energies)
    except AttributeError:
        E_disp_atom = np.zeros_like(E_atom)

    # --- Per-block breakdown (atomwise) ---
    # lists: length = num_blocks; each entry shape [N_atoms]
    blocks_raw = calc.block_atomic_energies_raw                         # unscaled heads
    blocks_scaled_no_shift = calc.block_atomic_energies_scaled_no_shift # Escale * head
    EshiftZ = calc.atom_energy_shift                                    # [N_atoms]

    # Stack to arrays with shape (num_blocks, N_atoms)
    B_raw = np.stack(blocks_raw, axis=0)
    B_scaled = np.stack(blocks_scaled_no_shift, axis=0)

    # Reconstruct NN per-atom energy from blocks: sum(blocks_scaled_no_shift) + shift
    E_atom_recon = B_scaled.sum(axis=0) + EshiftZ

    # Sanity check (small numerical differences possible)
    if not np.allclose(E_atom_recon, E_atom, atol=1e-6):
        print("[warn] per-atom reconstruction differs from E_atom (max abs diff = {:.3e})"
              .format(np.max(np.abs(E_atom_recon - E_atom))))

    # Molecular totals
    block_totals = B_scaled.sum(axis=1)        # per-block molecular (no shift)
    shift_total = EshiftZ.sum()
    E_NN_total = E_atom.sum()
    E_elec_total = E_elec_atom.sum()
    E_disp_total = E_disp_atom.sum()

    # Decomposition consistency
    if not np.allclose(E_tot, E_NN_total + E_elec_total + E_disp_total, atol=1e-5):
        print("[warn] total energy mismatch: E_tot={:.8f}, parts={:.8f}"
              .format(_as_scalar(E_tot), _as_scalar(E_NN_total + E_elec_total + E_disp_total)))

    # If dispersion is disabled, inferred dispersion should be ~0
    E_decomp_total = E_NN_total + E_elec_total
    E_disp_inferred = E_tot - E_decomp_total

    # Safe scalars for printing
    Etot   = _as_scalar(E_tot)
    ENN    = _as_scalar(E_NN_total)
    Eelec  = _as_scalar(E_elec_total)
    Edisp  = _as_scalar(E_disp_total)
    Bsum   = _as_scalar(block_totals.sum())
    Sshift = _as_scalar(shift_total)

    print("Frame {:3d} | Etot={: .6f}  ENN={: .6f}  Eelec={: .6f}  Edisp={: .6f}  sum(blocks)+shift={: .6f}"
          .format(i+1, Etot, ENN, Eelec, Edisp, Bsum + Sshift))

    # === Save everything ===
    npz_file = "mp2_perblock_analysis_{}.npz".format(i + 1)
    np.savez(
        npz_file,
        # geometry / identity
        Z=Z,
        R=R,
        # forces/charges/dipole
        F=F,
        Q=Q,
        D=mu,
        # total energies (molecular)
        E_tot=np.array([E_tot]),
        E_NN_total=np.array([E_NN_total]),
        E_elec_total=np.array([E_elec_total]),
        E_disp_total=np.array([E_disp_total]),
        # per-atom energies
        E_atom=E_atom,                 # NN short-range (scaled+shift)
        E_elec_atom=E_elec_atom,
        E_disp_atom=E_disp_atom,
        # per-block (atomwise)
        E_blocks_raw=B_raw,            # shape (num_blocks, N_atoms)
        E_blocks_scaled_no_shift=B_scaled,
        E_shift_atom=EshiftZ,          # [N_atoms]
        # reconstructions / checks
        E_atom_recon=E_atom_recon,
        block_totals=block_totals,     # per-module molecular contributions (no shift)
        shift_total=np.array([shift_total]),
        # representation if you need it
        act=act_frame,
        # optional inferred dispersion (use_dispersion=False scenario)
        E_disp_inferred=np.array([E_disp_inferred])
    )
    print("Frame {:3d}: saved to {}".format(i + 1, npz_file))

print("✅ All frames processed and saved.")

