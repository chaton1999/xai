#!/usr/bin/env python
# e_vs_mag_AB_overlay.py  (Python 3.5 compatible)
#
# Overlay A and B in the SAME 3x3 subplots:
#   x = total per-atom energy (E_atom + E_elec) [kcal/mol]
#   y = feature magnitude ||act[module, atom, :]||_2  (default: last module)
# Points are colored by signed distance from TS index (frame index).
#
# Your files are:
#   npz_1/mp2_perblock_analysis_*.npz
#   npz_1000000/mp2_perblock_analysis_*.npz

from __future__ import print_function
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mtick

try:
    from scipy.stats import pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ===============================
# Config
# ===============================
n_atoms = 9

# A/B folders (relative to where you run the script: mode_oh/)
A_DIR = "npz_1"
B_DIR = "npz_1000000"

# Transition-state frame index (0-based) for coloring
# For your NM scan with 1000 frames, TS is likely near the middle (499) unless you define it otherwise.
TS_INDEX = 499

CMAP_NAME = "RdBu_r"
EV_TO_KCAL = 23.06

# Padding for independent axes per subplot
PAD_FRAC_X = 0.10
PAD_FRAC_Y = 0.10
MIN_PAD_ABS = 1e-6

# Feature magnitude from which module?
USE_LAST_MODULE = True
MODULE_INDEX = -1  # last module

# Visual differentiation A vs B
A_MARKER = "o"
B_MARKER = "s"
A_ALPHA  = 0.65
B_ALPHA  = 0.65
A_SIZE   = 170
B_SIZE   = 120

# Tick formatter
num_formatter = mtick.FormatStrFormatter('%.2f')

# Save figure
SAVE = True
OUT_PNG = "AB_energy_vs_featmag_lastmodule_overlay.png"
DPI = 300

# ===============================
# Helpers
# ===============================
def extract_index(path):
    # mp2_perblock_analysis_1000.npz -> 1000
    m = re.search(r"analysis_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def padded_limits_and_ticks(vals, frac=0.10, min_pad_abs=1e-6):
    v = np.asarray(vals, dtype=float)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    width = vmax - vmin
    if width <= 0.0:
        pad = max(min_pad_abs, frac * max(1.0, abs(vmin)))
    else:
        pad = max(min_pad_abs, frac * width)
    lo = vmin - pad
    hi = vmax + pad
    mid = 0.5 * (lo + hi)
    return lo, mid, hi

def load_one(dirpath):
    """
    Returns:
      e_total_eV[atom] : list length Nframes
      feat_mag[atom]   : list length Nframes
      idxs             : list of frame indices (1-based from filenames), length Nframes
    """
    files = glob.glob("{0}/mp2_perblock_analysis_*.npz".format(dirpath))
    files = [f for f in files if extract_index(f) > 0]
    files = sorted(files, key=extract_index)

    if not files:
        raise IOError("No files found in {0} matching mp2_perblock_analysis_*.npz".format(dirpath))

    e_total = {i: [] for i in range(n_atoms)}
    feat_mag = {i: [] for i in range(n_atoms)}
    idxs = []

    for fn in files:
        data = np.load(fn)

        # required keys
        if "E_atom" not in data or "E_elec_atom" not in data or "act" not in data:
            raise KeyError(
                "{0}: missing keys. Have keys: {1}".format(fn, data.files)
            )

        atom_energy = data["E_atom"]  # (n_atoms,)
        elec_energy = data["E_elec_atom"]  # (n_atoms,)
        act = data["act"]             # (modules, atoms, feat_dim) typically

        # choose module for magnitude
        if USE_LAST_MODULE:
            mod = act[MODULE_INDEX]
        else:
            mod = act[MODULE_INDEX]

        if mod.shape[0] != n_atoms:
            raise ValueError("{0}: expected {1} atoms in act module, got {2}".format(fn, n_atoms, mod.shape[0]))

        for atom_idx in range(n_atoms):
            e_total[atom_idx].append(float(atom_energy[atom_idx] + elec_energy[atom_idx]))
            feat_mag[atom_idx].append(float(np.linalg.norm(mod[atom_idx])))

        idxs.append(extract_index(fn))

    return e_total, feat_mag, idxs

def safe_pearsonr(x, y):
    if not HAVE_SCIPY:
        return np.nan
    try:
        r, _ = pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return r
    except Exception:
        return np.nan

# ===============================
# Main
# ===============================
e_total_A, feat_mag_A, idxs_A = load_one(A_DIR)
e_total_B, feat_mag_B, idxs_B = load_one(B_DIR)

# Align frames by common indices (important if any missing)
common = sorted(set(idxs_A).intersection(set(idxs_B)))
if not common:
    raise IOError("No common frame indices between {0} and {1}".format(A_DIR, B_DIR))

# Build index->position maps
posA = {idx: i for i, idx in enumerate(idxs_A)}
posB = {idx: i for i, idx in enumerate(idxs_B)}

# signed distance from TS for coloring (based on common length index positions)
# Here we use the ORDER in `common` as the frame order.
N = len(common)
# If TS_INDEX is out of range, clamp to mid
if TS_INDEX < 0 or TS_INDEX >= N:
    TS_INDEX = N // 2

indices0 = np.arange(N)              # 0..N-1
svals = indices0 - TS_INDEX          # signed distance
vmax = float(max(abs(svals.min()), abs(svals.max())))
cmap = plt.get_cmap(CMAP_NAME)
norm = Normalize(vmin=-vmax, vmax=+vmax)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

print("A frames:", len(idxs_A), "B frames:", len(idxs_B), "Common:", N)
print("TS_INDEX (0-based in common frames):", TS_INDEX)

# ===============================
# Plot 3x3 overlay A+B
# ===============================
fig, axs = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

for atom_idx in range(n_atoms):
    row, col = divmod(atom_idx, 3)
    ax = axs[row, col]

    # collect aligned data for this atom
    xA_eV = []
    yA = []
    xB_eV = []
    yB = []
    for k, idx in enumerate(common):
        ia = posA[idx]
        ib = posB[idx]
        xA_eV.append(e_total_A[atom_idx][ia])
        yA.append(feat_mag_A[atom_idx][ia])
        xB_eV.append(e_total_B[atom_idx][ib])
        yB.append(feat_mag_B[atom_idx][ib])

    xA = np.asarray(xA_eV, dtype=float) * EV_TO_KCAL
    xB = np.asarray(xB_eV, dtype=float) * EV_TO_KCAL
    yA = np.asarray(yA, dtype=float)
    yB = np.asarray(yB, dtype=float)

    # shared axis limits using A ∪ B
    x_lo, x_mid, x_hi = padded_limits_and_ticks(np.concatenate([xA, xB]), frac=PAD_FRAC_X, min_pad_abs=MIN_PAD_ABS)
    y_lo, y_mid, y_hi = padded_limits_and_ticks(np.concatenate([yA, yB]), frac=PAD_FRAC_Y, min_pad_abs=MIN_PAD_ABS)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xticks([x_lo, x_mid, x_hi])
    ax.set_yticks([y_lo, y_mid, y_hi])
    ax.xaxis.set_major_formatter(num_formatter)
    ax.yaxis.set_major_formatter(num_formatter)

    # scatter, colored by signed distance
    ax.scatter(
        xA, yA,
        c=svals, cmap=cmap, norm=norm,
        s=A_SIZE, linewidths=0, alpha=A_ALPHA, marker=A_MARKER,
        label="A"
    )
    ax.scatter(
        xB, yB,
        c=svals, cmap=cmap, norm=norm,
        s=B_SIZE, linewidths=0, alpha=B_ALPHA, marker=B_MARKER,
        label="B"
    )

    rA = safe_pearsonr(xA_eV, yA)
    rB = safe_pearsonr(xB_eV, yB)

    ax.set_title("Atom {}  rA={:.2f} rB={:.2f}".format(atom_idx + 1, rA, rB), fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=8)
    ax.minorticks_off()

# legend once
axs[0, 0].legend(fontsize=18, frameon=False, loc="best")

# colorbar on the left with 0–TS–0 labeling (like your style)
signed_ticks = np.array([-vmax, -0.75*vmax, -0.5*vmax, -0.25*vmax, 0.0, 0.25*vmax, 0.5*vmax, 0.75*vmax, vmax])

def dist_label(x):
    return str(int(round(vmax - abs(x))))

tick_labels = [dist_label(x) for x in signed_ticks]

cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.95, pad=0.02, location="left")
cbar.set_label("Frames", fontsize=30)
cbar.set_ticks(signed_ticks)
cbar.set_ticklabels(tick_labels)
cbar.ax.tick_params(labelsize=26)

if SAVE:
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print("Saved:", OUT_PNG)

plt.show()

