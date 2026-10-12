#!/usr/bin/env python
# mag_ckpt_compare_module5.py  (Python 3.5 compatible)

from __future__ import print_function

import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

TAG_A = "ckpt_1000000"
TAG_B = "ckpt_1000000"

GLOB_A = (
    "nm_{0}/modes_{0}/mode_oh/npz_1/"
    "mp2_perblock_analysis_*.npz"
).format(TAG_A)

GLOB_B = (
    "nm_{0}/modes_{0}/mode_oh/npz_1000000/"
    "mp2_perblock_analysis_*.npz"
).format(TAG_B)

AMP    = 0.5
FRAMES = 1000
PATH   = "linear"   # "linear" or "sin"

MODULE_TO_PLOT = 5   # 1-based
N_ATOMS_PLOT   = 9

window_size   = 10
alpha_scatter = 0.20
s_scatter     = 35

A_COLOR = "tab:blue"
B_COLOR = "tab:red"

A_LABEL = "Ckpt_1"
B_LABEL = "Ckpt_10^6"

xlim = (-0.5, 0.5)
ylim = None

SAVE = True
DPI = 300

OUT_PNG = "magnitude_module{0}_A_vs_B.png".format(
    MODULE_TO_PLOT
)


# ============================================================
# MANUAL ATOM ORDER AND DISPLAY LABELS
# ============================================================
#
# Original / OLD atom ordering:
#
# Python index    old label
# -------------------------
#     0             O6
#     1             O8
#     2             H7
#     3             C3
#     4             C1
#     5             C0
#     6             H4
#     7             H2
#     8             H5
#
#
# Desired subplot arrangement:
#
# Atom C1     Atom C2     Atom H3
# Atom C4     Atom H5     Atom H6
# Atom O7     Atom H8     Atom O9
#
#
# Corresponding ORIGINAL atoms:
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
# Display labels corresponding EXACTLY to PLOT_ORDER
# ============================================================

MANUAL_LABELS_BY_PLOT_ORDER = [
    "C1", "C2", "H3",
    "C4", "H5", "H6",
    "O7", "H8", "O9"
]


# ============================================================
# OLD atom number -> NEW numerical atom number
# ============================================================
#
# Both are 1-based.
#
# Kept only for bookkeeping/checking.
# Subplot titles are controlled by
# MANUAL_LABELS_BY_PLOT_ORDER.
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
# Extract frame index from filename
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
# Mode coordinate
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

    # --------------------------------------------------------
    # File maps
    # --------------------------------------------------------

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
            "No common indices between A and B."
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
            "but common frames={1}. "
            "Using common indices.".format(
                FRAMES,
                len(common)
            )
        )


    # --------------------------------------------------------
    # X coordinate
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Initial arrays for shape checking
    # --------------------------------------------------------

    tmp_a = np.load(
        map_a[common[0]]
    )

    tmp_b = np.load(
        map_b[common[0]]
    )

    a0 = tmp_a["act"]
    b0 = tmp_b["act"]

    tmp_a.close()
    tmp_b.close()


    if (
        a0.ndim != 3
        or
        b0.ndim != 3
    ):

        raise ValueError(
            "Expected 3D act. "
            "Got A {0}, B {1}".format(
                a0.shape,
                b0.shape
            )
        )


    ma = a0.shape[0]
    mb = b0.shape[0]


    # --------------------------------------------------------
    # Detect possible extra module
    # --------------------------------------------------------

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
                "A {0} vs B {1}".format(
                    ma,
                    mb
                )
            )


    def maybe_drop(act, do_drop):

        if do_drop:
            return act[1:, :, :]

        return act


    a0 = maybe_drop(
        a0,
        drop_a
    )

    b0 = maybe_drop(
        b0,
        drop_b
    )


    if a0.shape != b0.shape:

        raise ValueError(
            "After alignment, shapes differ "
            "A {0} vs B {1}".format(
                a0.shape,
                b0.shape
            )
        )


    n_modules, n_atoms, feat_dim = a0.shape


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


    # --------------------------------------------------------
    # Check module
    # --------------------------------------------------------

    if (
        MODULE_TO_PLOT < 1
        or
        MODULE_TO_PLOT > n_modules
    ):

        raise ValueError(
            "Requested MODULE_TO_PLOT={0}, "
            "but available modules are 1..{1}".format(
                MODULE_TO_PLOT,
                n_modules
            )
        )


    mod_idx = MODULE_TO_PLOT - 1


    # --------------------------------------------------------
    # Number of atoms
    # --------------------------------------------------------

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
    # Calculate feature-vector magnitudes
    # ========================================================

    mag_a = np.zeros(
        (
            len(common),
            n_atoms
        ),
        dtype=float
    )

    mag_b = np.zeros(
        (
            len(common),
            n_atoms
        ),
        dtype=float
    )


    for i, idx in enumerate(common):

        data_a = np.load(
            map_a[idx]
        )

        data_b = np.load(
            map_b[idx]
        )

        act_a = data_a["act"]
        act_b = data_b["act"]


        act_a = maybe_drop(
            act_a,
            drop_a
        )

        act_b = maybe_drop(
            act_b,
            drop_b
        )


        # ----------------------------------------------------
        # Module selected above
        #
        # Shape:
        # (n_atoms, n_features)
        # ----------------------------------------------------

        Va = act_a[
            mod_idx,
            :,
            :
        ]

        Vb = act_b[
            mod_idx,
            :,
            :
        ]


        # ----------------------------------------------------
        # Magnitude of each atomic feature vector
        #
        # ||F_atom||
        # ----------------------------------------------------

        mag_a[i, :] = np.linalg.norm(
            Va,
            axis=1
        )

        mag_b[i, :] = np.linalg.norm(
            Vb,
            axis=1
        )


        data_a.close()
        data_b.close()


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 18)
    )

    axes = axes.flatten()


    # --------------------------------------------------------
    # General axis appearance
    # --------------------------------------------------------

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
    # Plot atoms using MANUAL ORDER
    # ========================================================

    for plot_pos, atom_idx in enumerate(
        PLOT_ORDER[:n_atoms_plot]
    ):

        ax = axes[
            plot_pos
        ]


        # ----------------------------------------------------
        # Chemical display label
        # ----------------------------------------------------
        #
        # This affects ONLY the title.
        #
        atom_label = (
            MANUAL_LABELS_BY_PLOT_ORDER[
                plot_pos
            ]
        )


        # ----------------------------------------------------
        # Numerical mapping retained if needed
        # ----------------------------------------------------

        old_atom_number = (
            atom_idx + 1
        )

        new_atom_number = (
            ATOM_LABEL_MAP[
                old_atom_number
            ]
        )


        # ----------------------------------------------------
        # Magnitudes
        #
        # atom_idx is still the ORIGINAL PhysNet atom index.
        # ----------------------------------------------------

        yA = mag_a[
            :,
            atom_idx
        ]

        yB = mag_b[
            :,
            atom_idx
        ]


        # ====================================================
        # Scatter
        # ====================================================

        ax.scatter(
            x_shift,
            yA,
            color=A_COLOR,
            alpha=alpha_scatter,
            s=s_scatter
        )

        ax.scatter(
            x_shift,
            yB,
            color=B_COLOR,
            alpha=alpha_scatter,
            s=s_scatter
        )


        # ====================================================
        # Running-average lines
        # ====================================================

        ax.plot(
            x_shift,
            running_average(
                yA,
                window_size
            ),
            color=A_COLOR,
            linewidth=2.8,
            label=A_LABEL
        )


        ax.plot(
            x_shift,
            running_average(
                yB,
                window_size
            ),
            color=B_COLOR,
            linewidth=2.8,
            label=B_LABEL
        )


        # ====================================================
        # SUBPLOT TITLE
        # ====================================================
        #
        # Result:
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
        # Limits
        # ====================================================

        if xlim is not None:

            ax.set_xlim(
                xlim[0],
                xlim[1]
            )


        if ylim is not None:

            ax.set_ylim(
                ylim[0],
                ylim[1]
            )


        # ====================================================
        # Position
        # ====================================================

        r = plot_pos // 3
        c = plot_pos % 3


        # ====================================================
        # X-axis
        # ====================================================

        if r == 2:

            ax.set_xlabel(
                "Mode coordinate A (Å)",
                fontsize=40
            )

        else:

            ax.set_xlabel("")

            ax.set_xticklabels([])


        # ====================================================
        # Y-axis
        # ====================================================

        if c == 0:

            ax.set_ylabel(
                r"$\|\mathbf{\mathcal{F}_{\rm atom}}\|$",
                fontsize=40
            )

        else:

            ax.set_ylabel("")

            ax.set_yticklabels([])


    # ========================================================
    # Hide unused panels
    # ========================================================

    for k in range(
        n_atoms_plot,
        9
    ):

        axes[k].axis(
            "off"
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
