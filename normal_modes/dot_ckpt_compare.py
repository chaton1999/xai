#!/usr/bin/env python
# dot_ckpt_compare.py  (Python 3.5 compatible)
#
# Compare features from ckpt_1 vs ckpt_1000000 along the SAME normal-mode scan:
#   for each frame j (1..1000): cosine( act_ckpt1[j], act_ckpt1000000[j] )
# Output:
#   5 figures (one per module), each a 3x3 panel for atoms 1..9.
#
# x-axis:
#   mode coordinate a in [-AMP, +AMP] (linear) or sin-path, consistent with normal_modes_one.py

from __future__ import print_function
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ========================== CONFIG ==========================
# Tags as produced by your workflow (folders nm_<TAG>/...)
TAG_A = "ckpt_1"
TAG_B = "ckpt_1"

# Normal-mode feature files
GLOB_A = "nm_{0}/modes_{0}/mode_oh/npz_1/mp2_perblock_analysis_*.npz".format(TAG_A)
GLOB_B = "nm_{0}/modes_{0}/mode_oh/npz_1000000/mp2_perblock_analysis_*.npz".format(TAG_B)

# Must match how trajectories were generated in normal_modes_one.py
AMP    = 0.5
FRAMES = 1000
PATH   = "linear"   # "linear" or "sin"

# Plot / smoothing
window_size   = 10
alpha_scatter = 0.20
colors  = ['tab:pink', 'tab:green', 'tab:blue', 'black', 'tab:red']
markers = ['o', 'o', 'o', 'o', 'o']

# Panel format
N_ATOMS_PLOT = 9   # 3x3 = first 9 atoms

# Optional x/y limits
xlim = (-0.5, 0.5)   # None to auto
ylim = (0.0, 1.05)

# Save?
SAVE = True
DPI = 300
OUT_PREFIX = "cosine_ckpt1_vs_ckpt1000000_modeoh"
# ============================================================


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
        # linspace(-amp, +amp, frames) with endpoint=True
        return -amp + (2.0 * amp) * float(j - 1) / float(frames - 1)
    elif path_type == "sin":
        # t = linspace(0, 2pi, frames, endpoint=False)
        t = 2.0 * np.pi * float(j - 1) / float(frames)
        return amp * np.sin(t)
    else:
        raise ValueError("Unknown PATH: {0}".format(path_type))


def load_file_map(glob_pat):
    """Return dict: idx -> filepath, only for idx >= 1."""
    files = glob.glob(glob_pat)
    d = {}
    for f in files:
        idx = extract_index(os.path.basename(f))
        if idx > 0:
            d[idx] = f
    return d


def main():
    # ---- Build index -> file maps for both checkpoints ----
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

    # ---- x-axis: mode coordinate a for each common frame index ----
    x_shift = np.array([alpha_from_index(j, AMP, FRAMES, PATH) for j in common], dtype=float)

    # ---- Load first pair to determine shapes ----
    a0 = np.load(map_a[common[0]])
    b0 = np.load(map_b[common[0]])

    if "act" not in a0 or "act" not in b0:
        raise KeyError("Missing 'act' key. A keys: {0}, B keys: {1}".format(a0.files, b0.files))

    act_a0 = a0["act"]
    act_b0 = b0["act"]

    if act_a0.ndim != 3 or act_b0.ndim != 3:
        raise ValueError("Expected 3D 'act'. Got A {0}, B {1}".format(act_a0.shape, act_b0.shape))

    # Remove first module consistently (same convention as your other scripts)
    # Handle cases where act may already exclude it.
    # We'll decide based on A, then enforce same for B.
    # If A has 6 blocks and you want 5, we drop [0].
    if act_a0.shape[0] >= 2:
        # we *assume* first module should be removed; but only if B matches similarly
        pass

    # Determine final module count after optional dropping:
    # If A has 6 and B has 6 -> drop both
    # If A has 5 and B has 5 -> keep both
    # If A has 6 and B has 5 -> drop A
    # If A has 5 and B has 6 -> drop B
    ma = act_a0.shape[0]
    mb = act_b0.shape[0]

    drop_a = False
    drop_b = False
    if ma == mb:
        # common case: both same
        # If you expect 5 blocks but data has 6, drop.
        # We'll drop if ma > 5 (your PhysNet has 5 blocks; sometimes file includes an extra 0th module)
        if ma > 5:
            drop_a = True
            drop_b = True
    else:
        # mismatch: make them match by dropping the extra first module on the larger one
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

    # ---- Preallocate cosine array: (n_frames_common, n_modules, n_atoms) ----
    cosine_sim = np.zeros((len(common), n_modules, n_atoms), dtype=float)

    # ---- Compute frame-by-frame cosine similarity between checkpoints ----
    for i, idx in enumerate(common):
        arr_a = np.load(map_a[idx])
        arr_b = np.load(map_b[idx])

        act_a = arr_a["act"]
        act_b = arr_b["act"]

        if act_a.ndim != 3 or act_b.ndim != 3:
            raise ValueError("Frame {0}: act not 3D (A {1}, B {2})".format(idx, act_a.shape, act_b.shape))

        act_a = maybe_drop(act_a, drop_a)
        act_b = maybe_drop(act_b, drop_b)

        if act_a.shape != (n_modules, n_atoms, feat_dim):
            raise ValueError("Frame {0}: A shape {1} != expected {2}".format(
                idx, act_a.shape, (n_modules, n_atoms, feat_dim)))
        if act_b.shape != (n_modules, n_atoms, feat_dim):
            raise ValueError("Frame {0}: B shape {1} != expected {2}".format(
                idx, act_b.shape, (n_modules, n_atoms, feat_dim)))

        a_norm = normalize_rows(act_a)
        b_norm = normalize_rows(act_b)

        # cosine per (module, atom)
        cosine_sim[i, :, :] = np.sum(a_norm * b_norm, axis=-1)

    # ---- Plot: 5 figures (modules), each 3x3 atoms ----
    n_atoms_plot = min(N_ATOMS_PLOT, n_atoms)
    if n_atoms_plot != 9:
        print("[WARN] 3x3 expects 9 atoms, but n_atoms={0}. Plotting first {1} atoms.".format(n_atoms, n_atoms_plot))

    for mod in range(n_modules):
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

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

        for k in range(n_atoms_plot, 9):
            axes[k].axis("off")

        fig.suptitle("Cosine(ckpt_1 vs ckpt_1000000) — Module {0}".format(mod + 1), fontsize=34)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out = "{0}_module_{1}.png".format(OUT_PREFIX, mod + 1)
        if SAVE:
            plt.savefig(out, dpi=DPI, bbox_inches="tight")
            print("Saved:", out)
        plt.show()


if __name__ == "__main__":
    main()

