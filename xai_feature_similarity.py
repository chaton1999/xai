
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3: Atom-wise M5 output-feature cosine similarity.

Original sources:
  project_feature/modular_energy/barrier_height/
    elec_barrier_optimizations/model.ckpt-N/
    t1_barrier_optimizations/model.ckpt-N/

Staged outputs:
  XAI/data/feature_similarity/ref_N/
    true.dat
    false.dat

Use EXTRACT=True once, then EXTRACT=False for plotting.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import (
    LogLocator,
    LogFormatterMathtext,
    AutoMinorLocator,
)


class FeatureSimilarity:

    # Old zero-based indices in desired C1...O9 order
    ATOM_ORDER = [5, 4, 7, 3, 6, 8, 0, 2, 1]

    ATOM_LABELS = [
        "C1", "C2", "H3",
        "C4", "H5", "H6",
        "O7", "H8", "O9",
    ]

    COLORS = [
        "#440154", "#46327E", "#365C8D",
        "#2F6C8E", "#21918C", "#2FB47C",
        "#5DC863", "#95D840", "#FDE725",
    ]

    LINESTYLES = [
        "-", ":", "--",
        "-.", "-", ":",
        "--", "-.", "-",
    ]

    def __init__(self, xai_root=None):

        self.root = Path(
            xai_root or Path.home() / "XAI"
        ).expanduser().resolve()

        self.source_dir = (
            self.root
            / "project_feature"
            / "modular_energy"
            / "barrier_height"
        )

        self.data_dir = (
            self.root / "data" / "feature_similarity"
        )

        self.model_roots = {
            "true": self.source_dir / "elec_barrier_optimizations",
            "false": self.source_dir / "t1_barrier_optimizations",
        }

        self.data = {}

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    @staticmethod
    def find_checkpoints(root, npz_name):

        root = Path(root)

        if not root.is_dir():
            raise FileNotFoundError(f"Missing directory: {root}")

        found = {}

        for directory in root.iterdir():

            if not directory.is_dir():
                continue

            match = re.fullmatch(
                r"model\.ckpt-(\d+)",
                directory.name
            )

            if match is None:
                continue

            checkpoint = int(match.group(1))
            npz_path = directory / npz_name

            if npz_path.is_file():
                found[checkpoint] = npz_path

        if not found:
            raise RuntimeError(
                f"No checkpoint NPZ files found in {root}"
            )

        return dict(sorted(found.items()))

    # ========================================================
    # LOAD FEATURE MATRIX
    # ========================================================

    @staticmethod
    def load_features(
        path,
        feature_key="act_output",
        module_index=4,
        n_atoms=9,
    ):

        with np.load(path, allow_pickle=True) as npz:

            if feature_key not in npz.files:
                raise KeyError(
                    f"{feature_key!r} missing in {path}\n"
                    f"Available keys: {npz.files}"
                )

            array = np.asarray(npz[feature_key])

        if array.dtype == object:
            array = np.asarray(array.tolist())

        if array.ndim == 3:
            if not 0 <= module_index < array.shape[0]:
                raise ValueError(
                    f"Module index {module_index} invalid "
                    f"for shape {array.shape} in {path}"
                )
            array = array[module_index]

        if array.ndim != 2:
            raise ValueError(
                f"Expected 2D atom features, got "
                f"{array.shape} in {path}"
            )

        if array.shape[0] == n_atoms:
            features = array
        elif array.shape[1] == n_atoms:
            features = array.T
        else:
            raise ValueError(
                f"Cannot identify atom axis for "
                f"{array.shape} in {path}"
            )

        return np.asarray(features, dtype=float)

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_per_atom(features, reference):

        if features.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch: {features.shape} "
                f"vs {reference.shape}"
            )

        dot = np.sum(features * reference, axis=1)

        norms = (
            np.linalg.norm(features, axis=1)
            * np.linalg.norm(reference, axis=1)
        )

        result = np.full(len(dot), np.nan)

        np.divide(
            dot,
            norms,
            out=result,
            where=norms > 0,
        )

        return np.clip(result, -1.0, 1.0)

    # ========================================================
    # EXTRACT ONE MODEL
    # ========================================================

    def extract_model(
        self,
        mode,
        reference_checkpoint=1000000,
        npz_name="opt_opt_irc_reactant_analysis.npz",
        feature_key="act_output",
        module_index=4,
    ):

        files = self.find_checkpoints(
            self.model_roots[mode],
            npz_name,
        )

        if reference_checkpoint not in files:
            raise FileNotFoundError(
                f"Reference checkpoint {reference_checkpoint} "
                f"not found for electrostatics={mode}"
            )

        reference = self.load_features(
            files[reference_checkpoint],
            feature_key,
            module_index,
        )

        rows = []

        for checkpoint, path in files.items():

            features = self.load_features(
                path,
                feature_key,
                module_index,
            )

            similarity = self.cosine_per_atom(
                features,
                reference,
            )

            row = {"checkpoint": checkpoint}

            for label, atom_idx in zip(
                self.ATOM_LABELS,
                self.ATOM_ORDER,
            ):
                row[label] = float(similarity[atom_idx])

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("checkpoint")

        destination = (
            self.data_dir
            / f"ref_{reference_checkpoint}"
        )
        destination.mkdir(parents=True, exist_ok=True)

        out = destination / f"{mode}.dat"

        df.to_csv(
            out,
            sep=" ",
            index=False,
            float_format="%.12g",
        )

        self.data[mode] = df

        print(
            f"E_elec={mode}: {len(df)} checkpoints -> {out}"
        )

        return df

    # ========================================================
    # LOAD STAGED DATA
    # ========================================================

    def load_model(self, mode, reference_checkpoint):

        path = (
            self.data_dir
            / f"ref_{reference_checkpoint}"
            / f"{mode}.dat"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Staged file missing: {path}\n"
                "Run with EXTRACT=True first."
            )

        df = pd.read_csv(path, sep=r"\s+")

        self.data[mode] = df

        return df

    # ========================================================
    # MAIN ENTRY POINT
    # ========================================================

    def run(
        self,
        extract=False,
        mode="both",
        reference_checkpoint=1000000,
        show_data=False,
        show_plot=True,
        save_plot=True,
        show_legend=True,
        show_grid=False,
        show_vertical_lines=True,
        xlim=(1, 1000000),
        ylim=(0.80, 1.01),
        figsize=(14, 7),
        linewidth=3.5,
        alpha=0.95,
        label_size=20,
        tick_size=17,
        title_size=20,
        legend_size=17,
        spine_width=2.5,
        major_tick_length=9,
        minor_tick_length=5,
        major_tick_width=2.5,
        minor_tick_width=1.8,
        legend_ncol=3,
        legend_x=0.5,
        legend_y=1.08,
        legend_handlelength=2.7,
        left=0.12,
        right=0.97,
        bottom=0.15,
        top=0.76,
        wspace=0.08,
        vertical_true=(18, 212),
        vertical_false=(18, 321),
        vertical_color="black",
        vertical_width=1.5,
        vertical_style="--",
        vertical_alpha=0.65,
        show_panel_labels=True,
        dpi=300,
        output_name="feature_similarity_m5",
        npz_name="opt_opt_irc_reactant_analysis.npz",
        feature_key="act_output",
        module_index=4,
    ):

        if mode not in ("both", "true", "false"):
            raise ValueError(
                "mode must be 'both', 'true', or 'false'"
            )

        modes = (
            ["true", "false"]
            if mode == "both"
            else [mode]
        )

        # ----------------------------------------------------
        # Extract or load
        # ----------------------------------------------------

        for current_mode in modes:

            if extract:
                df = self.extract_model(
                    mode=current_mode,
                    reference_checkpoint=reference_checkpoint,
                    npz_name=npz_name,
                    feature_key=feature_key,
                    module_index=module_index,
                )
            else:
                df = self.load_model(
                    current_mode,
                    reference_checkpoint,
                )

            if show_data:
                try:
                    from IPython.display import display
                    display(df)
                except ImportError:
                    print(df)

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            len(modes),
            figsize=figsize,
            sharey=(len(modes) == 2),
            squeeze=False,
        )

        axes = axes.ravel()

        legend_handles = []

        for panel_index, (ax, current_mode) in enumerate(
            zip(axes, modes)
        ):

            df = self.data[current_mode]

            for i, atom in enumerate(self.ATOM_LABELS):

                line, = ax.plot(
                    df["checkpoint"],
                    df[atom],
                    color=self.COLORS[i],
                    linestyle=self.LINESTYLES[i],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=atom,
                    zorder=2,
                )

                if panel_index == 0:
                    legend_handles.append(line)

            # ------------------------------------------------
            # Vertical checkpoints
            # ------------------------------------------------

            if show_vertical_lines:

                checkpoints = (
                    vertical_true
                    if current_mode == "true"
                    else vertical_false
                )

                for checkpoint in checkpoints:
                    ax.axvline(
                        checkpoint,
                        color=vertical_color,
                        linestyle=vertical_style,
                        linewidth=vertical_width,
                        alpha=vertical_alpha,
                        zorder=3,
                    )

            # ------------------------------------------------
            # Axis limits
            # ------------------------------------------------

            this_xlim = (
                xlim.get(current_mode)
                if isinstance(xlim, dict)
                else xlim
            )

            this_ylim = (
                ylim.get(current_mode)
                if isinstance(ylim, dict)
                else ylim
            )

            ax.set_xscale("log")

            if this_xlim is not None:
                ax.set_xlim(*this_xlim)

            if this_ylim is not None:
                ax.set_ylim(*this_ylim)

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            ax.set_xlabel(
                r"Transfer Learning Steps $N$",
                fontsize=label_size,
            )

            if panel_index == 0:
                ax.set_ylabel(
                    r"$\mathcal{P}^{\alpha}$",
                    fontsize=label_size,
                )

            ax.set_title(
                rf"$E_{{\mathrm{{elec}}}} = "
                rf"\mathrm{{{current_mode.capitalize()}}}$",
                fontsize=title_size,
            )

            # ------------------------------------------------
            # Tick formatting
            # ------------------------------------------------

            ax.xaxis.set_major_locator(
                LogLocator(base=10, numticks=8)
            )

            ax.xaxis.set_major_formatter(
                LogFormatterMathtext(base=10)
            )

            ax.xaxis.set_minor_locator(
                LogLocator(
                    base=10,
                    subs=np.arange(2, 10),
                    numticks=100,
                )
            )

            ax.yaxis.set_minor_locator(
                AutoMinorLocator(4)
            )

            ax.tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                labelsize=tick_size,
                width=major_tick_width,
                length=major_tick_length,
            )

            ax.tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                width=minor_tick_width,
                length=minor_tick_length,
            )

            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)

            ax.grid(show_grid, alpha=0.15)

            # ------------------------------------------------
            # Panel labels
            # ------------------------------------------------

            if show_panel_labels:
                ax.text(
                    0.95,
                    0.05,
                    chr(ord("A") + panel_index),
                    transform=ax.transAxes,
                    fontsize=title_size,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )

        # ----------------------------------------------------
        # Figure layout
        # ----------------------------------------------------

        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
        )

        # ----------------------------------------------------
        # Legend above panels
        # ----------------------------------------------------

        if show_legend:

            fig.legend(
                legend_handles,
                self.ATOM_LABELS,
                loc="upper center",
                bbox_to_anchor=(legend_x, legend_y),
                ncol=legend_ncol,
                fontsize=legend_size,
                frameon=False,
                handlelength=legend_handlelength,
                columnspacing=1.8,
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if save_plot:

            out_dir = (
                self.data_dir
                / f"ref_{reference_checkpoint}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            name = (
                f"{output_name}_{mode}"
                f"_ref_{reference_checkpoint}"
            )

            fig.savefig(
                out_dir / f"{name}.png",
                dpi=dpi,
                bbox_inches="tight",
            )

            fig.savefig(
                out_dir / f"{name}.pdf",
                bbox_inches="tight",
            )

            print(f"Saved figures in: {out_dir}")

        # Avoid duplicate Jupyter display
        if show_plot:
            plt.show()

        plt.close(fig)

        # Return datasets, not Matplotlib figure
        if mode == "both":
            return {
                "true": self.data["true"],
                "false": self.data["false"],
            }

        return self.data[mode]

