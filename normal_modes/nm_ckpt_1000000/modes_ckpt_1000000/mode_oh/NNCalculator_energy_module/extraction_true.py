#!/usr/bin/env python3
"""
Export all NeuralNetwork variables from a PhysNet checkpoint.

This script:
  - Builds a NeuralNetwork (same hyperparams & scope as in NNCalculator)
  - Restores weights using nn.restore(sess, ckpt)
  - Exports all variables in nn.variables to <out>.npz
  - Writes <out>.json with TF names and shapes

Usage example:
  python extraction_w_b.py \
      --ckpt all_runs/model.ckpt-1/models/model.ckpt-1 \
      --out feature_data/1_physnet_vars.npz \
      --F 128 --K 64 --num-blocks 5 --sr-cut 6.0 --lr-cut 10.0
"""

import json
import argparse
import numpy as np
import tensorflow as tf
import os

try:
    from NNCalculator_energy_module.neural_network.NeuralNetwork import NeuralNetwork
except Exception as e:
    print("ERROR: Could not import NeuralNetwork from NNCalculator_energy_module.")
    print("Adjust import path.")
    raise e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Checkpoint path (e.g. models/model.ckpt-1000000)")
    ap.add_argument("--out", required=True,
                    help="Output .npz filename (e.g. feature_data/1000000_physnet_vars.npz)")

    # PhysNet hyperparameters
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--sr-cut", type=float, default=6.0)
    ap.add_argument("--lr-cut", type=float, default=None)
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--num-residual-atomic", type=int, default=2)
    ap.add_argument("--num-residual-interaction", type=int, default=3)
    ap.add_argument("--num-residual-output", type=int, default=1)

    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--scope", type=str, default="neural_network")

    args = ap.parse_args()

    # ---- FIXED FLAGS ----
    use_electrostatic = True
    use_dispersion = False   # <--- DISABLED

    dtype_tf = tf.float32 if args.dtype == "float32" else tf.float64

    # ---------------- Build model ----------------
    nn = NeuralNetwork(
        F=args.F,
        K=args.K,
        sr_cut=args.sr_cut,
        lr_cut=args.lr_cut,
        num_blocks=args.num_blocks,
        num_residual_atomic=args.num_residual_atomic,
        num_residual_interaction=args.num_residual_interaction,
        num_residual_output=args.num_residual_output,
        use_electrostatic=use_electrostatic,
        use_dispersion=use_dispersion,
        dtype=dtype_tf,
        scope=args.scope
    )

    var_arrays = {}
    meta = {
        "ckpt": args.ckpt,
        "F": args.F,
        "K": args.K,
        "sr_cut": args.sr_cut,
        "lr_cut": args.lr_cut,
        "num_blocks": args.num_blocks,
        "num_residual_atomic": args.num_residual_atomic,
        "num_residual_interaction": args.num_residual_interaction,
        "num_residual_output": args.num_residual_output,
        "use_electrostatic": use_electrostatic,
        "use_dispersion": use_dispersion,
        "dtype": args.dtype,
        "scope": args.scope,
        "notes": "Electrostatics enabled. Dispersion DISABLED."
    }

    # ---------------- Restore weights ----------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nn.restore(sess, args.ckpt)
        print("[ok] Restored:", args.ckpt)

        nn_vars = nn.variables

        for short_name, var in nn_vars.items():
            name_clean = short_name.replace(":", "_")
            value = sess.run(var)

            var_arrays[name_clean] = value
            meta[name_clean] = {
                "tf_short_name": short_name,
                "tf_full_name": args.scope + "/" + short_name,
                "shape": value.shape
            }

    # ---------------- Save outputs ----------------
    out_dir = os.path.dirname(args.out)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.out, **var_arrays)

    meta_path = args.out.replace(".npz", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("[ok] Saved arrays to:", args.out)
    print("[ok] Saved metadata to:", meta_path)


if __name__ == "__main__":
    main()

