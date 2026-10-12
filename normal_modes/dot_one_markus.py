#!/usr/bin/env python
# dot_ckpt_compare_onefig.py  (Python 3.5 compatible)

from __future__ import print_function

import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FixedLocator


# ============================================================
# CONFIG
# ============================================================

TAG_A = "ckpt_1"
TAG_B = "ckpt_1"

GLOB_A = (
    "nm_{0}/modes_{0}/mode_oh/npz_1/"
    "mp2_perblock_analysis_*.npz"
).format(TAG_A)

GLOB_B = (
    "nm_{0}/modes_{0}/mode_oh/npz_1000000/"
    "mp2_perblock_analysis_*.npz"
).format(TAG_B)

AMP    = -0.5
FRAMES = 1000
PATH   = "linear"       # "linear" or "sin"

window_size   = 10
alpha_scatter = 0.80

N_ATOMS_PLOT = 9

xlim = (-0.5, 0.5)
ylim = (0.78, 1.0)

SAVE = True
DPI = 300

OUT_PNG = (
    "cosine_ckpt1_vs_ckpt1000000_"
    "modeoh_allmodules_onefig.png"
)


# ============================================================
# MANUAL ATOM ORDER AND DISPLAY LABELS
# ============================================================
#
# ORIGINAL / OLD atom identities:
#
# old Python index     old chemical label
# ---------------------------------------
#       0                    O6
#       1                    O8
#       2                    H7
#       3                    C3
#       4                    C1
#       5                    C0
#       6                    H4
#       7                    H2
#       8                    H5
#
#
# Desired displayed arrangement:
#
# Atom C1     Atom C2     Atom H3
# Atom C4     Atom H5     Atom H6
# Atom O7     Atom H8     Atom O9
#
#
# Therefore the ORIGINAL atoms are arranged as:
#
# C0          C1          H2
# C3          H4          H5
# O6          H7          O8
#
# ============================================================

PLOT_ORDER = [
    5, 4, 7,       # C1, C2, H3
    3, 6, 8,       # C4, H5, H6
    0, 2, 1        # O7, H8, O9
]


# ============================================================
# Manual displayed labels
# ============================================================
#
# IMPORTANT:
# These labels correspond exactly to the order in PLOT_ORDER.
#
MANUAL_LABELS_BY_PLOT_ORDER = [
    "C1", "C2", "H3",
    "C4", "H5", "H6",
    "O7", "H8", "O9"
]


# ============================================================
# OLD atom number -> NEW atom number
# ============================================================
#
# Both are 1-based.
#
# This is retained only for bookkeeping/checking.
# It does NOT determine the displayed subplot title anymore.
#
ATOM_LABEL_MAP = {
    1: 7,      # O6 -> O7
    2: 9,      # O8 -> O9
    3: 8,      # H7 -> H8
    4: 4,      # C3 -> C4
    5: 2,      # C1 -> C2
    6: 1,      # C0 -> C1
    7: 5,      # H4 -> H5
    8: 3,      # H2 -> H3
    9: 6,      # H5 -> H6
}


# ============================================================
# OLD atom index -> subplot position
# ============================================================

PLOT_POS_BY_OLD_IDX = {
    old_idx: plot_pos
    for plot_pos, old_idx in enumerate(PLOT_ORDER)
}


# ============================================================
# Plotting style for modules
# ============================================================

SCATTER_MODULES = [0, 4]
LINE_MODULES    = [1, 2, 3]

MOD_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
}

SCATTER_MARKER = "o"


# ============================================================
# Normalize feature vectors
# ============================================================

def normalize_rows(V, eps=1e-12):

    norms = np.linalg.norm(
        V,
        axis=-1,
        keepdims=True
    )

    return np.where(
        norms > eps,
        V / norms,
        0.0
    )


# ============================================================
# Running average
# ============================================================

def running_average(y, window):

    if window < 2:
        return y

    pad = window // 2

    y_pad = np.pad(
        y,
        (pad, pad),
        mode="edge"
    )

    kernel = (
        np.ones(window, dtype=float)
        /
        float(window)
    )

    y_smooth = np.convolve(
        y_pad,
        kernel,
        mode="valid"
    )

    if len(y_smooth) > len(y):

        diff = len(y_smooth) - len(y)

        y_smooth = y_smooth[
            diff // 2:
            diff // 2 + len(y)
        ]

    return y_smooth


# ============================================================
# Extract geometry/frame index
# ============================================================

def extract_index(fname):

    m = re.search(
        r"analysis_(\d+)\.npz$",
        fname
    )

    if m:
        return int(m.group(1))

    return -1


# ============================================================
# Generate mode coordinate
# ============================================================

def alpha_from_index(
    j,
    amp,
    frames,
    path_type
):

    if path_type == "linear":

        return (
            -amp
            +
            (2.0 * amp)
            *
            float(j - 1)
            /
            float(frames - 1)
        )

    elif path_type == "sin":

        t = (
            2.0
            *
            np.pi
            *
            float(j - 1)
            /
            float(frames)
        )

        return amp * np.sin(t)

    else:

        raise ValueError(
            "Unknown PATH: {0}".format(
                path_type
            )
        )


# ============================================================
# Load file map
# ============================================================

def load_file_map(glob_pat):

    files = glob.glob(
        glob_pat
    )

    d = {}

    for f in files:

        idx = extract_index(
            os.path.basename(f)
        )

        if idx > 0:

            d[idx] = f

    return d


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load checkpoint file maps
    # ========================================================

    map_a = load_file_map(
        GLOB_A
    )

    map_b = load_file_map(
        GLOB_B
    )


    if not map_a:

        raise IOError(
            "No files found for A with pattern: "
            "{0}".format(GLOB_A)
        )


    if not map_b:

        raise IOError(
            "No files found for B with pattern: "
            "{0}".format(GLOB_B)
        )


    common = sorted(
        set(map_a.keys()).intersection(
            set(map_b.keys())
        )
    )


    if not common:

        raise IOError(
            "No common indices between A and B. "
            "Check paths/tags."
        )


    print(
        "A glob:",
        GLOB_A
    )

    print(
        "B glob:",
        GLOB_B
    )

    print(
        "A frames:",
        len(map_a),
        "B frames:",
        len(map_b),
        "Common:",
        len(common)
    )


    if len(common) != FRAMES:

        print(
            "[WARN] Expected {0} frames, "
            "but common frames = {1}. "
            "Using common indices.".format(
                FRAMES,
                len(common)
            )
        )


    # ========================================================
    # Generate x-axis mode coordinate
    # ========================================================

    x_shift = np.array(
        [
            alpha_from_index(
                j,
                AMP,
                FRAMES,
                PATH
            )
            for j in common
        ],
        dtype=float
    )


    # ========================================================
    # Read first files to determine dimensions
    # ========================================================

    a0 = np.load(
        map_a[common[0]]
    )

    b0 = np.load(
        map_b[common[0]]
    )


    if "act" not in a0 or "act" not in b0:

        raise KeyError(
            "Missing 'act' key. "
            "A keys: {0}, B keys: {1}".format(
                a0.files,
                b0.files
            )
        )


    act_a0 = a0["act"]
    act_b0 = b0["act"]


    if (
        act_a0.ndim != 3
        or
        act_b0.ndim != 3
    ):

        raise ValueError(
            "Expected 3D 'act'. "
            "Got A {0}, B {1}".format(
                act_a0.shape,
                act_b0.shape
            )
        )


    ma = act_a0.shape[0]
    mb = act_b0.shape[0]


    # ========================================================
    # Handle possible extra first module
    # ========================================================

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

            raise ValueError(
                "Module dimension mismatch "
                "A {0} vs B {1} "
                "(cannot reconcile)".format(
                    ma,
                    mb
                )
            )


    def maybe_drop(act, do_drop):

        if do_drop:

            return act[1:, :, :]

        return act


    act_a0 = maybe_drop(
        act_a0,
        drop_a
    )

    act_b0 = maybe_drop(
        act_b0,
        drop_b
    )


    if act_a0.shape != act_b0.shape:

        raise ValueError(
            "After module alignment, "
            "shapes still differ: "
            "A {0} vs B {1}".format(
                act_a0.shape,
                act_b0.shape
            )
        )


    n_modules, n_atoms, feat_dim = (
        act_a0.shape
    )


    print(
        "Using modules:",
        n_modules,
        "atoms:",
        n_atoms,
        "feat_dim:",
        feat_dim
    )

    print(
        "Drop first module? A:",
        drop_a,
        "B:",
        drop_b
    )


    # ========================================================
    # Calculate cosine similarity
    # ========================================================
    #
    # Shape:
    #
    # [frame, module, atom]
    #
    cosine_sim = np.zeros(
        (
            len(common),
            n_modules,
            n_atoms
        ),
        dtype=float
    )


    for i, idx in enumerate(common):

        arr_a = np.load(
            map_a[idx]
        )

        arr_b = np.load(
            map_b[idx]
        )


        act_a = maybe_drop(
            arr_a["act"],
            drop_a
        )

        act_b = maybe_drop(
            arr_b["act"],
            drop_b
        )


        # Normalize each atomic feature vector
        a_norm = normalize_rows(
            act_a
        )

        b_norm = normalize_rows(
            act_b
        )


        # Dot product of normalized vectors
        # = cosine similarity
        cosine_sim[i, :, :] = np.sum(
            a_norm * b_norm,
            axis=-1
        )


        arr_a.close()
        arr_b.close()


    # ========================================================
    # Number of atoms to plot
    # ========================================================

    n_atoms_plot = min(
        N_ATOMS_PLOT,
        n_atoms
    )


    if n_atoms_plot != 9:

        print(
            "[WARN] 3x3 expects 9 atoms, "
            "but n_atoms={0}. "
            "Plotting first {1} atoms.".format(
                n_atoms,
                n_atoms_plot
            )
        )


    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 18)
    )

    axes = axes.flatten()


    # ========================================================
    # General axis style
    # ========================================================

    for ax in axes:

        for spine in ax.spines.values():

            spine.set_linewidth(
                4.5
            )


        ax.tick_params(
            axis="x",
            width=3.5,
            length=7,
            labelsize=40,
            rotation=60
        )


        ax.tick_params(
            axis="y",
            width=3.5,
            length=7,
            labelsize=40
        )


    # ========================================================
    # Plot atoms in MANUAL order
    # ========================================================

    for plot_pos, atom_idx in enumerate(
        PLOT_ORDER[:n_atoms_plot]
    ):

        ax = axes[
            plot_pos
        ]


        # ====================================================
        # DISPLAYED CHEMICAL ATOM LABEL
        # ====================================================
        #
        # This ONLY controls the title.
        #
        # The actual cosine data continue to use atom_idx,
        # i.e. the original PhysNet atom ordering.
        #
        atom_label = (
            MANUAL_LABELS_BY_PLOT_ORDER[
                plot_pos
            ]
        )


        # ====================================================
        # Keep numerical mappings available if needed
        # ====================================================

        old_atom_number = (
            atom_idx + 1
        )

        new_atom_number = (
            ATOM_LABEL_MAP[
                old_atom_number
            ]
        )


        # ====================================================
        # Scatter modules M1 and M5
        # ====================================================

        for mod in SCATTER_MODULES:

            if mod >= n_modules:

                continue


            y = cosine_sim[
                :,
                mod,
                atom_idx
            ]


            ax.scatter(
                x_shift,
                y,
                color=MOD_COLORS[mod],
                alpha=alpha_scatter,
                marker=SCATTER_MARKER,
                s=45,
                label="M{0}".format(
                    mod + 1
                )
            )


        # ====================================================
        # Smoothed line modules M2, M3, M4
        # ====================================================

        for mod in LINE_MODULES:

            if mod >= n_modules:

                continue


            y = cosine_sim[
                :,
                mod,
                atom_idx
            ]


            ax.plot(
                x_shift,
                running_average(
                    y,
                    window_size
                ),
                color=MOD_COLORS[mod],
                linewidth=4.6,
                linestyle="--",
                label="M{0}".format(
                    mod + 1
                )
            )


        # ====================================================
        # SUBPLOT TITLE
        # ====================================================
        #
        # Gives:
        #
        # Atom C1   Atom C2   Atom H3
        # Atom C4   Atom H5   Atom H6
        # Atom O7   Atom H8   Atom O9
        #
        ax.set_title(
            "Atom {0}".format(
                atom_label
            ),
            fontsize=30
        )


        # ====================================================
        # Axis ranges
        # ====================================================

        ax.set_ylim(
            ylim[0],
            ylim[1]
        )


        if xlim is not None:

            ax.set_xlim(
                xlim[0],
                xlim[1]
            )


        # ====================================================
        # Y-axis ticks
        # ====================================================

        ax.yaxis.set_major_locator(
            FixedLocator(
                [0.8, 0.9, 1.0]
            )
        )

        ax.yaxis.set_minor_locator(
            MultipleLocator(
                0.02
            )
        )

        ax.minorticks_on()


        ax.tick_params(
            axis="y",
            which="minor",
            width=3.0,
            length=8
        )


        ax.tick_params(
            axis="y",
            which="major",
            width=3.0,
            length=12
        )


        # ====================================================
        # Row/column position
        # ====================================================

        r = plot_pos // 3
        c = plot_pos % 3


        # ====================================================
        # X-axis label only on bottom row
        # ====================================================

        if r == 2:

            ax.set_xlabel(
                "Mode coordinate a (Å)",
                fontsize=40
            )

        else:

            ax.set_xlabel("")

            ax.set_xticklabels([])


        # ====================================================
        # Y-axis label only on first column
        # ====================================================

        if c == 0:

            ax.set_ylabel(
                r"$\mathcal{P}$",
                fontsize=40
            )

        else:

            ax.set_ylabel("")

            ax.set_yticklabels([])


    # ========================================================
    # Hide unused axes if fewer than 9 atoms
    # ========================================================

    for k in range(
        n_atoms_plot,
        9
    ):

        axes[k].axis(
            "off"
        )


    # ========================================================
    # Legend
    # ========================================================
    #
    # With 9 atoms, axes[8] corresponds to Atom O9.
    #
    handles, labels = (
        axes[8].get_legend_handles_labels()
    )


    if handles:

        seen = {}

        new_h = []
        new_l = []


        for h, l in zip(
            handles,
            labels
        ):

            if l not in seen:

                seen[l] = True

                new_h.append(h)
                new_l.append(l)


        axes[2].legend(
            new_h,
            new_l,
            fontsize=28,
            ncol=3,
            frameon=False,
            loc="best",
            markerscale=2.6
        )


    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(
        rect=[
            0,
            0.03,
            1,
            0.95
        ]
    )


    # ========================================================
    # Save
    # ========================================================

    if SAVE:

        plt.savefig(
            OUT_PNG,
            dpi=DPI,
            bbox_inches="tight"
        )

        print(
            "Saved:",
            OUT_PNG
        )


    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
