#!/usr/bin/env python
# dot_ckpt_compare_onefig.py  (Python 3.5 compatible)
#
# Same cosine computation as dot_ckpt_compare.py, but plot EVERYTHING in ONE figure:
#   - 3x3 atoms (1..9)
#   - Module 1 and 5: scatter only (same marker, different colors)
#   - Module 2/3/4: dashed lines (different colors)
#
# Results are identical; only plotting changes.

from __future__ import print_function
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FixedLocator, AutoMinorLocator

# ========================== CONFIG ==========================
TAG_A = "ckpt_1"
TAG_B = "ckpt_1"

GLOB_A = "nm_{0}/modes_{0}/mode_oh/npz_1/mp2_perblock_analysis_*.npz".format(TAG_A)
GLOB_B = "nm_{0}/modes_{0}/mode_oh/npz_1000000/mp2_perblock_analysis_*.npz".format(TAG_B)

AMP    = -0.5
FRAMES = 1000
PATH   = "linear"   # "linear" or "sin"

window_size   = 10
alpha_scatter = 0.80

# Panel format
N_ATOMS_PLOT = 9   # first 9 atoms

xlim = (-0.5, 0.5)   # None to auto
ylim = (0.78, 1.0)

SAVE = True
DPI = 300
OUT_PNG = "cosine_ckpt1_vs_ckpt1000000_modeoh_allmodules_onefig.png"
# ============================================================

# --- plotting style for modules ---
# module indices are 0..4
SCATTER_MODULES = [0, 4]   # M1 and M5 scatter
LINE_MODULES    = [1, 2, 3] # M2-4 dashed lines

# Colors per module (distinct)
MOD_COLORS = {
    0: "tab:blue",    # M1
    1: "tab:orange",  # M2
    2: "tab:green",   # M3
    3: "tab:red",     # M4
    4: "tab:purple",  # M5
}
SCATTER_MARKER = "o"

def normalize_rows(V, eps=1e-12):
    norms = np.linalg.norm(V, axis=-1, keepdims=True)
    return np.where(norms > eps, V / norms, 0.0)

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
        raise IOError("No common indices between A and B. Check paths/tags.")

    print("A glob:", GLOB_A)
    print("B glob:", GLOB_B)
    print("A frames:", len(map_a), "B frames:", len(map_b), "Common:", len(common))
    if len(common) != FRAMES:
        print("[WARN] Expected {0} frames, but common frames = {1}. Using common indices.".format(FRAMES, len(common)))

    x_shift = np.array([alpha_from_index(j, AMP, FRAMES, PATH) for j in common], dtype=float)

    a0 = np.load(map_a[common[0]])
    b0 = np.load(map_b[common[0]])

    if "act" not in a0 or "act" not in b0:
        raise KeyError("Missing 'act' key. A keys: {0}, B keys: {1}".format(a0.files, b0.files))

    act_a0 = a0["act"]
    act_b0 = b0["act"]

    if act_a0.ndim != 3 or act_b0.ndim != 3:
        raise ValueError("Expected 3D 'act'. Got A {0}, B {1}".format(act_a0.shape, act_b0.shape))

    ma = act_a0.shape[0]
    mb = act_b0.shape[0]

    drop_a = False
    drop_b = False
    if ma == mb:
        if ma > 5:
            drop_a = True
            drop_b = True
    else:
        if ma == mb + 1:
            drop_a = True
        elif mb == ma + 1:
            drop_b = True
        else:
            raise ValueError("Module dimension mismatch A {0} vs B {1} (cannot reconcile)".format(ma, mb))

    def maybe_drop(act, do_drop):
        return act[1:, :, :] if do_drop else act

    act_a0 = maybe_drop(act_a0, drop_a)
    act_b0 = maybe_drop(act_b0, drop_b)

    if act_a0.shape != act_b0.shape:
        raise ValueError("After module alignment, shapes still differ: A {0} vs B {1}".format(act_a0.shape, act_b0.shape))

    n_modules, n_atoms, feat_dim = act_a0.shape
    print("Using modules:", n_modules, "atoms:", n_atoms, "feat_dim:", feat_dim)
    print("Drop first module? A:", drop_a, "B:", drop_b)

    cosine_sim = np.zeros((len(common), n_modules, n_atoms), dtype=float)

    for i, idx in enumerate(common):
        arr_a = np.load(map_a[idx])
        arr_b = np.load(map_b[idx])

        act_a = maybe_drop(arr_a["act"], drop_a)
        act_b = maybe_drop(arr_b["act"], drop_b)

        a_norm = normalize_rows(act_a)
        b_norm = normalize_rows(act_b)

        cosine_sim[i, :, :] = np.sum(a_norm * b_norm, axis=-1)

    # ---------- ONE FIGURE: 3x3 atoms, all modules overlaid ----------
    n_atoms_plot = min(N_ATOMS_PLOT, n_atoms)
    if n_atoms_plot != 9:
        print("[WARN] 3x3 expects 9 atoms, but n_atoms={0}. Plotting first {1} atoms.".format(n_atoms, n_atoms_plot))

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(4.5)
        ax.tick_params(axis="x",width=3.5, length=7, labelsize=40, rotation=60)
        ax.tick_params(axis="y",width=3.5, length=7, labelsize=40)

    for atom_idx in range(n_atoms_plot):
        ax = axes[atom_idx]

        # Module 1 & 5 scatter (no lines)
        for mod in SCATTER_MODULES:
            if mod >= n_modules:
                continue
            y = cosine_sim[:, mod, atom_idx]
            ax.scatter(x_shift, y, color=MOD_COLORS[mod], alpha=alpha_scatter,
                       marker=SCATTER_MARKER, s=45, label="M{0}".format(mod + 1))

        # Modules 2-4 dashed lines (running average)
        for mod in LINE_MODULES:
            if mod >= n_modules:
                continue
            y = cosine_sim[:, mod, atom_idx]
            ax.plot(x_shift, running_average(y, window_size),
                    color=MOD_COLORS[mod], linewidth=4.6, linestyle="--",
                    label="M{0}".format(mod + 1))

        ax.set_title("Atom {0}".format(atom_idx + 1), fontsize=30)
        ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])


        # --- Y ticks: force 0.8, 0.9, 1.0 (and show minors in-between) ---
        ax.yaxis.set_major_locator(FixedLocator([0.8, 0.9, 1.0]))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))   # 0.02 spacing: 0.82,0.84,...0.98
        ax.minorticks_on()
        ax.tick_params(axis="y", which="minor", width=3.0, length=8)
        ax.tick_params(axis="y", which="major", width=3.0, length=12)

        # --- X ticks (optional): majors every 0.1, minors every 0.05 ---
#        ax.xaxis.set_major_locator(MultipleLocator(0.1))
#        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
#        ax.tick_params(axis="x", which="minor", width=2.0, length=4)


        r = atom_idx // 3
        c = atom_idx % 3
        if r == 2:
            ax.set_xlabel("Mode coordinate a (Å)", fontsize=40)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(r"$\mathcal{P}$", fontsize=40)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    for k in range(n_atoms_plot, 9):
        axes[k].axis("off")

    # legend only once (first panel), deduplicate labels
    handles, labels = axes[8].get_legend_handles_labels()
    if handles:
        seen = {}
        new_h = []
        new_l = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                new_h.append(h)
                new_l.append(l)
        axes[8].legend(new_h, new_l, fontsize=28, ncol=3, frameon=False, loc="best", markerscale=2.6)

#    fig.suptitle("Cosine(ckpt_1 vs ckpt_1000000) — all modules in one figure", fontsize=34)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if SAVE:
        plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
        print("Saved:", OUT_PNG)

    plt.show()

if __name__ == "__main__":
    main()

