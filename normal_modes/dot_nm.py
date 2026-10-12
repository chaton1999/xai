#!/usr/bin/env python
# dot_np.py  (Python 3.5 compatible)
#
# Cosine similarity (normalized dot product) of NM-scan feature vectors vs optimized reference.
# Output: 5 figures (one per module), each a 3x3 panel for atoms 1..9.

from __future__ import print_function
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ========================== CONFIG ==========================
# Which checkpoint/tag to analyze:
#   tag folder names are nm_<TAG>/... and your ref file is opt_rs_ckpt_<CKPT>.xyz.npz
CKPT = "1000000"          # "1" or "1000000"
TAG  = "ckpt_1000000"      # "ckpt_1" or "ckpt_1000000"

# Optimized reference feature file (in current directory)
OPT_FILE = "opt_rs_ckpt_{0}.xyz.npz".format(CKPT)

# Normal mode feature files (1000 frames)
NM_GLOB = "nm_{0}/modes_{0}/mode_oh/npz/mp2_perblock_analysis_*.npz".format(TAG)

# Match your normal_modes_one.py settings used to generate the mode trajectories
AMP    = 0.5
FRAMES = 1000
PATH   = "linear"   # "linear" or "sin"

# Plot / smoothing
window_size   = 10
alpha_scatter = 0.20
colors  = ['tab:pink', 'tab:green', 'tab:blue', 'black', 'tab:red']
markers = ['o', 'o', 'o', 'o', 'o']

# If you want to clamp x-range, set tuple like (-0.5, 0.5), else None
xlim = (-0.5, 0.5)

# y-limits
ylim = (0.0, 1.05)

# Save images?
SAVE = True
OUT_PREFIX = "cosine_vs_opt_nm_{0}_ckpt_{1}".format(TAG, CKPT)
DPI = 300
# ============================================================


def normalize_rows(V, eps=1e-12):
    """Normalize along last axis; near-zero vectors become 0."""
    norms = np.linalg.norm(V, axis=-1, keepdims=True)
    return np.where(norms > eps, V / norms, 0.0)


def running_average(y, window):
    """Centered running average with edge padding; output same length."""
    if window < 2:
        return y
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    y_smooth = np.convolve(y_pad, kernel, mode='valid')
    if len(y_smooth) > len(y):
        diff = len(y_smooth) - len(y)
        y_smooth = y_smooth[diff//2 : diff//2 + len(y)]
    return y_smooth


def extract_index(fname):
    """mp2_perblock_analysis_1000.npz -> 1000"""
    m = re.search(r"analysis_(\d+)\.npz$", fname)
    if m:
        return int(m.group(1))
    return -1


def alpha_from_index(j, amp, frames, path_type):
    """
    Convert 1-based frame index j (1..frames) to mode coordinate 'a'
    consistent with normal_modes_one.py export_mode_traj().
    """
    if path_type == "linear":
        # linspace(-amp, +amp, frames) with endpoint included
        return -amp + (2.0 * amp) * float(j - 1) / float(frames - 1)
    elif path_type == "sin":
        # t = linspace(0, 2pi, frames, endpoint=False); a = amp*sin(t)
        t = 2.0 * np.pi * float(j - 1) / float(frames)
        return amp * np.sin(t)
    else:
        raise ValueError("Unknown PATH: {0}".format(path_type))


def main():
    # ---------- Locate files ----------
    if not os.path.isfile(OPT_FILE):
        raise IOError("Missing optimized reference file: {0}".format(OPT_FILE))

    nm_files = glob.glob(NM_GLOB)
    nm_files = sorted(nm_files, key=lambda p: extract_index(os.path.basename(p)))
    nm_files = [f for f in nm_files if extract_index(os.path.basename(f)) > 0]

    if not nm_files:
        raise IOError("No normal-mode files found with pattern: {0}".format(NM_GLOB))

    # indices from filenames (so x-axis matches even if missing frames)
    idx_list = [extract_index(os.path.basename(f)) for f in nm_files]

    print("Reference:", OPT_FILE)
    print("NM glob:", NM_GLOB)
    print("Found {0} NM feature files.".format(len(nm_files)))
    if len(nm_files) != FRAMES:
        print("[WARN] Expected {0} frames, found {1}. Using filename indices for x-axis.".format(FRAMES, len(nm_files)))

    # x-axis = mode coordinate a (Å)
    x_shift = np.array([alpha_from_index(j, AMP, FRAMES, PATH) for j in idx_list], dtype=float)

    # ---------- Load optimized reference ----------
    ref_act = np.load(OPT_FILE)["act"]
    if ref_act.ndim != 3:
        raise ValueError("Expected ref 'act' to be 3D (modules, atoms, feat), got {0}".format(ref_act.shape))

    # Remove the first module (same pattern as your previous scripts)
    ref_act = ref_act[1:, :, :]
    n_modules, n_atoms, feat_dim = ref_act.shape
    print("Dims — modules kept: {0}, atoms: {1}, feat_dim: {2}".format(n_modules, n_atoms, feat_dim))

    ref_norm = normalize_rows(ref_act)

    # ---------- Preallocate ----------
    cosine_sim = np.zeros((len(nm_files), n_modules, n_atoms), dtype=float)

    # ---------- Compute cosines ----------
    for i, fpath in enumerate(nm_files):
        arr = np.load(fpath)

        if "act" not in arr:
            raise KeyError("{0}: no key 'act'. Keys: {1}".format(fpath, arr.files))

        act = arr["act"]
        if act.ndim != 3:
            raise ValueError("{0}: expected act 3D, got {1}".format(fpath, act.shape))

        # allow optional extra first module
        if act.shape[0] == n_modules + 1:
            act = act[1:, :, :]
        elif act.shape[0] != n_modules:
            raise ValueError("{0}: module mismatch. act.shape[0]={1}, expected {2} or {3}".format(
                fpath, act.shape[0], n_modules, n_modules + 1))

        if act.shape[1] != n_atoms or act.shape[2] != feat_dim:
            raise ValueError("{0}: shape mismatch. got {1}, expected ({2},{3},{4})".format(
                fpath, act.shape, n_modules, n_atoms, feat_dim))

        act_norm = normalize_rows(act)
        cosine_sim[i, :, :] = np.sum(act_norm * ref_norm, axis=-1)

    # ---------- Plot: 5 figures (modules), each 3x3 atoms ----------
    n_atoms_plot = 9
    if n_atoms < 9:
        n_atoms_plot = n_atoms
    if n_atoms != 9:
        print("[WARN] Requested 3x3 (9 atoms) but n_atoms={0}. Plotting first {1} atoms.".format(n_atoms, n_atoms_plot))

    for mod in range(n_modules):
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        # style
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            ax.tick_params(width=2.5, length=7, labelsize=26)

        color  = colors[mod % len(colors)]
        marker = markers[mod % len(markers)]

        for atom_idx in range(n_atoms_plot):
            ax = axes[atom_idx]
            y = cosine_sim[:, mod, atom_idx]

            ax.scatter(x_shift, y, color=color, alpha=alpha_scatter, marker=marker, s=50)
            ax.plot(x_shift, running_average(y, window_size), color=color, linewidth=2.8)

            ax.set_title("Atom {0}".format(atom_idx + 1), fontsize=30)
            ax.set_ylim(ylim[0], ylim[1])
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])

            r = atom_idx // 3
            c = atom_idx % 3

            if r == 2:
                ax.set_xlabel("Mode coordinate a (Å)", fontsize=30)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(r"$\mathcal{p}$", fontsize=30)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

        # hide unused panels (if n_atoms_plot < 9)
        for k in range(n_atoms_plot, 9):
            axes[k].axis("off")

        fig.suptitle("Cosine vs optimized reference — Module {0} ({1}, ckpt {2})".format(mod + 1, TAG, CKPT),
                     fontsize=34)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out = "{0}_module_{1}.png".format(OUT_PREFIX, mod + 1)
        if SAVE:
            plt.savefig(out, dpi=DPI, bbox_inches="tight")
            print("Saved:", out)
        plt.show()


if __name__ == "__main__":
    main()

