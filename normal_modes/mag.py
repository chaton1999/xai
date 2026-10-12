#!/usr/bin/env python
# mag_ckpt_compare_module5.py  (Python 3.5 compatible)
#
# Plot feature-vector magnitude per atom for TWO datasets (A,B) on the SAME 3x3 grid,
# using ONLY module 5 (after optional dropping of first module).
#
# y = || act[module, atom, :] ||_2
# Two curves: A and B, same subplot.

from __future__ import print_function
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ========================== CONFIG ==========================
TAG_A = "ckpt_1000000"
TAG_B = "ckpt_1000000"

GLOB_A = "nm_{0}/modes_{0}/mode_oh/npz_1/mp2_perblock_analysis_*.npz".format(TAG_A)
GLOB_B = "nm_{0}/modes_{0}/mode_oh/npz_1000000/mp2_perblock_analysis_*.npz".format(TAG_B)

AMP    = 0.5
FRAMES = 1000
PATH   = "linear"   # "linear" or "sin"

# choose module number in human terms (module 5)
MODULE_TO_PLOT = 5   # 1-based

# 3x3 for atoms 1..9
N_ATOMS_PLOT = 9

# smoothing / style
window_size   = 10
alpha_scatter = 0.20
s_scatter     = 35

# A vs B styling
A_COLOR = "tab:blue"
B_COLOR = "tab:red"
A_LABEL = "Ckpt_1"
B_LABEL = "Ckpt_10^6"

# Optional axis limits
xlim = (-0.5, 0.5)   # None to auto
ylim = None          # e.g. (0, 5) or None for auto

SAVE = True
DPI = 300
OUT_PNG = "magnitude_module{0}_A_vs_B.png".format(MODULE_TO_PLOT)
# ============================================================

def running_average(y, window):
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
    m = re.search(r"analysis_(\d+)\.npz$", fname)
    if m:
        return int(m.group(1))
    return -1

def alpha_from_index(j, amp, frames, path_type):
    if path_type == "linear":
        return -amp + (2.0 * amp) * float(j - 1) / float(frames - 1)
    elif path_type == "sin":
        t = 2.0 * np.pi * float(j - 1) / float(frames)
        return amp * np.sin(t)
    else:
        raise ValueError("Unknown PATH: {0}".format(path_type))

def load_file_map(glob_pat):
    files = glob.glob(glob_pat)
    d = {}
    for f in files:
        idx = extract_index(os.path.basename(f))
        if idx > 0:
            d[idx] = f
    return d

def main():
    map_a = load_file_map(GLOB_A)
    map_b = load_file_map(GLOB_B)

    if not map_a:
        raise IOError("No files found for A with pattern: {0}".format(GLOB_A))
    if not map_b:
        raise IOError("No files found for B with pattern: {0}".format(GLOB_B))

    common = sorted(set(map_a.keys()).intersection(set(map_b.keys())))
    if not common:
        raise IOError("No common indices between A and B.")

    print("A glob:", GLOB_A)
    print("B glob:", GLOB_B)
    print("A frames:", len(map_a), "B frames:", len(map_b), "Common:", len(common))
    if len(common) != FRAMES:
        print("[WARN] Expected {0} frames, but common frames={1}. Using common indices.".format(FRAMES, len(common)))

    x_shift = np.array([alpha_from_index(j, AMP, FRAMES, PATH) for j in common], dtype=float)

    # ---- inspect shapes to decide whether to drop first module ----
    a0 = np.load(map_a[common[0]])["act"]
    b0 = np.load(map_b[common[0]])["act"]

    if a0.ndim != 3 or b0.ndim != 3:
        raise ValueError("Expected 3D act. Got A {0}, B {1}".format(a0.shape, b0.shape))

    ma = a0.shape[0]
    mb = b0.shape[0]

    drop_a = False
    drop_b = False
    if ma == mb:
        if ma > 5:   # if extra leading module exists
            drop_a = True
            drop_b = True
    else:
        if ma == mb + 1:
            drop_a = True
        elif mb == ma + 1:
            drop_b = True
        else:
            raise ValueError("Module dimension mismatch A {0} vs B {1}".format(ma, mb))

    def maybe_drop(act, do_drop):
        return act[1:, :, :] if do_drop else act

    a0 = maybe_drop(a0, drop_a)
    b0 = maybe_drop(b0, drop_b)
    if a0.shape != b0.shape:
        raise ValueError("After alignment, shapes differ A {0} vs B {1}".format(a0.shape, b0.shape))

    n_modules, n_atoms, feat_dim = a0.shape
    print("Using modules:", n_modules, "atoms:", n_atoms, "feat_dim:", feat_dim)
    print("Drop first module? A:", drop_a, "B:", drop_b)

    if MODULE_TO_PLOT < 1 or MODULE_TO_PLOT > n_modules:
        raise ValueError("Requested MODULE_TO_PLOT={0}, but available modules are 1..{1}".format(MODULE_TO_PLOT, n_modules))
    mod_idx = MODULE_TO_PLOT - 1  # 0-based

    n_atoms_plot = min(N_ATOMS_PLOT, n_atoms)
    if n_atoms_plot != 9:
        print("[WARN] 3x3 expects 9 atoms, but n_atoms={0}. Plotting first {1} atoms.".format(n_atoms, n_atoms_plot))

    # ---- collect magnitudes: (n_frames, n_atoms) for module mod_idx ----
    mag_a = np.zeros((len(common), n_atoms), dtype=float)
    mag_b = np.zeros((len(common), n_atoms), dtype=float)

    for i, idx in enumerate(common):
        act_a = np.load(map_a[idx])["act"]
        act_b = np.load(map_b[idx])["act"]

        act_a = maybe_drop(act_a, drop_a)
        act_b = maybe_drop(act_b, drop_b)

        # act_* shape: (n_modules, n_atoms, feat_dim)
        Va = act_a[mod_idx, :, :]
        Vb = act_b[mod_idx, :, :]

        mag_a[i, :] = np.linalg.norm(Va, axis=1)
        mag_b[i, :] = np.linalg.norm(Vb, axis=1)

    # ---- Plot 3x3 with both A and B in each subplot ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(4.5)
        ax.tick_params(axis="x",width=3.5, length=7, labelsize=40, rotation=60)
        ax.tick_params(axis="y",width=3.5, length=7, labelsize=40)


    for atom_idx in range(n_atoms_plot):
        ax = axes[atom_idx]

        yA = mag_a[:, atom_idx]
        yB = mag_b[:, atom_idx]

        ax.scatter(x_shift, yA, color=A_COLOR, alpha=alpha_scatter, s=s_scatter)
        ax.scatter(x_shift, yB, color=B_COLOR, alpha=alpha_scatter, s=s_scatter)

        ax.plot(x_shift, running_average(yA, window_size), color=A_COLOR, linewidth=2.8, label=A_LABEL)
        ax.plot(x_shift, running_average(yB, window_size), color=B_COLOR, linewidth=2.8, label=B_LABEL)

        ax.set_title("Atom {0}".format(atom_idx + 1), fontsize=30)

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        r = atom_idx // 3
        c = atom_idx % 3
        if r == 2:
            ax.set_xlabel("Mode coordinate a (Å)", fontsize=40)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(r"$\||\mathbf{\mathcal{F}_{\rm atom}}\||$", fontsize=40)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # hide unused axes
    for k in range(n_atoms_plot, 9):
        axes[k].axis("off")

    # legend (put on last used axis)
    #lr = (n_atoms_plot - 1) // 3
    #lc = (n_atoms_plot - 1) % 3
    #axes[lr * 3 + lc].legend(fontsize=30, frameon=False, loc="best",markerscale=2.5)

   # fig.suptitle("Feature magnitude, Module {0}: A vs B".format(MODULE_TO_PLOT), fontsize=34)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if SAVE:
        plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
        print("Saved:", OUT_PNG)
    plt.show()


if __name__ == "__main__":
    main()

