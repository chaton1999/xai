#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import numpy as np
from os.path import splitext
from ase.io import read, write
from ase.optimize import BFGS
from NNCalculator.NNCalculator import NNCalculator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)              # <- used in output naming
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=1e-4)
    args = p.parse_args()

    filename, ext = splitext(os.path.basename(args.input))

    atoms = read(args.input)

    # center
    atoms.set_positions(atoms.get_positions() - np.mean(atoms.get_positions(), axis=0))

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

    out_xyz = "opt_{0}_{1}.xyz".format(filename, args.tag)
    out_log = "opt_{0}_{1}.log".format(filename, args.tag)
    out_trj = "opt_{0}_{1}.traj".format(filename, args.tag)

    print("Optimizing:", args.input, "with", args.checkpoint, "tag", args.tag)
    dyn = BFGS(atoms, logfile=out_log, trajectory=out_trj)
    dyn.run(fmax=args.fmax)
    write(out_xyz, atoms)
    print("Wrote:", out_xyz)


if __name__ == "__main__":
    main()

