#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import numpy as np
from ase.io import read, write
from ase.vibrations import Vibrations
from NNCalculator.NNCalculator import NNCalculator


def is_complex(x):
    # works for numpy scalars too
    try:
        return np.iscomplexobj(x) and (np.abs(np.imag(x)) > 0.0)
    except Exception:
        return False


def freq_to_label(f):
    """
    Make a safe label for filenames.
    If imaginary/complex: use abs(real/imag) and suffix 'i'
    """
    if is_complex(f):
        val = float(np.abs(f))
        return "{:08.2f}i".format(val)
    else:
        return "{:08.2f}".format(float(f))


def export_mode_traj(ref_atoms, mode_flat, amp_ang, n_frames, out_path, path_type):
    N = len(ref_atoms)
    v = mode_flat.reshape((N, 3)).copy()

    #vn = np.linalg.norm(v)
    #if vn < 1e-20:
    #    return
    #v /= vn

    per_atom = np.linalg.norm(v, axis=1)
    m = float(np.max(per_atom))
    if m < 1e-20:
        return
    v /= m


    pos0 = ref_atoms.get_positions()
    frames = []

    if path_type == "linear":
        alphas = np.linspace(-amp_ang, amp_ang, int(n_frames))
    elif path_type == "sin":
        t = np.linspace(0.0, 2.0*np.pi, int(n_frames), endpoint=False)
        alphas = amp_ang * np.sin(t)
    else:
        raise ValueError("Unknown path_type: {}".format(path_type))

    for a in alphas:
        at = ref_atoms.copy()
        at.set_positions(pos0 + a * v)
        frames.append(at)

    write(out_path, frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="optimized xyz")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--charge", type=float, default=0.0)

    p.add_argument("--freq_min", type=float, default=20.0)
    p.add_argument("--freq_max", type=float, default=None)

    p.add_argument("--amp", type=float, default=0.25)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--path", type=str, default="linear", choices=["linear", "sin"])
    p.add_argument("--export_modes", type=str, default="all")  # "all" or "0,1,2"

    args = p.parse_args()

    atoms = read(args.input)

    outdir = os.path.abspath("nm_{0}".format(args.tag))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    calc = NNCalculator(
        checkpoint=args.checkpoint,
        atoms=atoms,
        charge=args.charge,
        F=128, K=64, num_blocks=5,
        num_residual_atomic=2,
        num_residual_interaction=3,
        num_residual_output=1,
        sr_cut=10.0,
        use_electrostatic=True,
        use_dispersion=False
    )
    atoms.set_calculator(calc)

    print("Vibrations for:", args.input, "tag", args.tag)

    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        vib = Vibrations(atoms, name="vib_" + args.tag)
        vib.run()

        freqs_cm = np.array(vib.get_frequencies())  # may include complex entries
        nm = 3 * len(atoms)

        modes = np.zeros((nm, nm), float)
        for i in range(nm):
            modes[i] = vib.get_mode(i).reshape(-1)

        vib.summary()

        np.savez(args.tag + "_vib_raw.npz", freqs_cm=freqs_cm, modes=modes)

        # filter using magnitude, so complex is fine
        mag = np.abs(freqs_cm)
        mask = (mag >= float(args.freq_min))
        if args.freq_max is not None:
            mask &= (mag <= float(args.freq_max))

        idx = np.where(mask)[0]
        freqs_f = freqs_cm[idx]
        modes_f = modes[idx]

        np.savez(args.tag + "_vib_filtered.npz", idx=idx, freqs_cm=freqs_f, modes=modes_f)

        print("Kept modes:", len(idx), "/", len(freqs_cm))

        vib.clean()

        if args.export_modes == "all":
            export_list = list(range(len(freqs_f)))
        else:
            export_list = [int(x) for x in args.export_modes.split(",") if x.strip()]

        modes_dir = "modes_" + args.tag
        if not os.path.isdir(modes_dir):
            os.makedirs(modes_dir)

        for k in export_list:
            f = freqs_f[k]
            ftag = freq_to_label(f)
            out_traj = os.path.join(
                modes_dir,
                "mode_{0:03d}_f_{1}.traj".format(k, ftag)
            )
            export_mode_traj(atoms, modes_f[k], args.amp, args.frames, out_traj, args.path)

        print("Wrote mode trajectories in:", os.path.abspath(modes_dir))

    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()

