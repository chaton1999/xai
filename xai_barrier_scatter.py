
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2: Correlation between H-transfer barrier height and energy MAE.

Input:
    ~/XAI/data/combined_9columns.dat

Modes:
    "both"  : Electrostatics True and False
    "true"  : Electrostatics True only
    "false" : Electrostatics False only

No PhysNet calculations are performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext, AutoMinorLocator


class BarrierMAEScatter:

    def __init__(self, xai_root=None):

        self.root = Path(
            xai_root or Path.home() / "XAI"
        ).expanduser().resolve()

        self.data_file = (
            self.root / "data" / "combined_9columns.dat"
        )

        self.output_dir = self.root / "data"

        self.df = None

    # ========================================================
    # LOAD DATA
    # ========================================================

    def load(self):

        if not self.data_file.is_file():
            raise FileNotFoundError(
                f"Cannot find data file:\n{self.data_file}"
            )

        df = pd.read_csv(
            self.data_file,
            sep=r"\s+"
        )

        required = [
            "checkpoint",
            "Eb_true",
            "EMAE_true",
            "Eb_false",
            "EMAE_false",
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        self.df = df.sort_values("checkpoint")

        return self.df

    # ========================================================
    # PLOT
    # ========================================================

    def plot(
        self,
        mode="both",
        show_plot=True,
        save_plot=True,
        output_name="mae_vs_barrier",
        figsize=None,
        cmap="plasma",
        xlim=None,
        ylim=None,
        marker_size=32,
        highlight_size=140,
        show_highlights=True,
        show_colorbar=True,
        show_grid=False,
        label_size=20,
        tick_size=16,
        title_size=19,
        spine_width=2.0,
        major_tick_length=9,
        minor_tick_length=5,
        major_tick_width=2.0,
        minor_tick_width=1.5,
        dpi=300,

        # Figure layout
        left=0.12,
        right=0.86,
        bottom=0.16,
        top=0.90,
        wspace=0.10,

        # Colorbar settings
        colorbar_pad=0.025,
        colorbar_fraction=0.045,
        colorbar_label_size=20,

        # Highlight checkpoints
        highlight_true=(18, 212),
        highlight_false=(18, 321),
    ):

        if mode not in ("both", "true", "false"):
            raise ValueError(
                "mode must be 'both', 'true', or 'false'"
            )

        df = self.load() if self.df is None else self.df

        modes = (
            ["true", "false"]
            if mode == "both"
            else [mode]
        )

        ncols = len(modes)

        if figsize is None:
            figsize = (14, 6) if ncols == 2 else (8, 6)

        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=figsize,
            sharey=(ncols == 2),
            squeeze=False,
        )

        axes = axes.ravel()

        # ----------------------------------------------------
        # Shared checkpoint normalization
        # ----------------------------------------------------

        valid_checkpoints = df.loc[
            df["checkpoint"] > 0, "checkpoint"
        ]

        if valid_checkpoints.empty:
            raise ValueError("No positive checkpoints found.")

        norm = LogNorm(
            vmin=1,
            vmax=max(1e6, float(valid_checkpoints.max()))
        )

        colormap = plt.get_cmap(cmap)

        # ----------------------------------------------------
        # Draw panels
        # ----------------------------------------------------

        for ax, current_mode in zip(axes, modes):

            xcol = f"Eb_{current_mode}"
            ycol = f"EMAE_{current_mode}"

            data = df[
                ["checkpoint", xcol, ycol]
            ].dropna()

            # Logarithmic axis requires positive y
            data = data[
                (data["checkpoint"] > 0)
                & (data[ycol] > 0)
                & np.isfinite(data[xcol])
                & np.isfinite(data[ycol])
            ]

            scatter = ax.scatter(
                data[xcol],
                data[ycol],
                c=data["checkpoint"],
                cmap=colormap,
                norm=norm,
                s=marker_size,
                edgecolors="none",
                zorder=2,
            )

            # ------------------------------------------------
            # Highlight selected checkpoints
            # ------------------------------------------------

            if show_highlights:

                marked = (
                    highlight_true
                    if current_mode == "true"
                    else highlight_false
                )

                symbols = ["x", "d"]

                for checkpoint, symbol in zip(marked, symbols):

                    point = data.loc[
                        data["checkpoint"] == checkpoint
                    ]

                    if point.empty:
                        print(
                            f"Warning: checkpoint {checkpoint} "
                            f"not found for E_elec={current_mode}"
                        )
                        continue

                    x = float(point[xcol].iloc[0])
                    y = float(point[ycol].iloc[0])

                    # Draw black outline
                    ax.scatter(
                        [x],
                        [y],
                        marker=symbol,
                        s=highlight_size,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=2.0,
                        zorder=6,
                    )

            # ------------------------------------------------
            # Axis labels
            # ------------------------------------------------

            ax.set_title(
                rf"$E_{{\mathrm{{elec}}}} = "
                rf"\mathrm{{{current_mode.capitalize()}}}$",
                fontsize=title_size,
            )

            ax.set_xlabel(
                r"$E_b$ (kcal/mol)",
                fontsize=label_size,
            )

            if ax is axes[0]:
                ax.set_ylabel(
                    r"MAE$(E)$ (kcal/mol)",
                    fontsize=label_size,
                )

            ax.set_yscale("log")

            # ------------------------------------------------
            # Independent axis ranges
            # ------------------------------------------------

            current_xlim = (
                xlim.get(current_mode)
                if isinstance(xlim, dict)
                else xlim
            )

            current_ylim = (
                ylim.get(current_mode)
                if isinstance(ylim, dict)
                else ylim
            )

            if current_xlim is not None:
                ax.set_xlim(*current_xlim)

            if current_ylim is not None:
                ax.set_ylim(*current_ylim)

            # ------------------------------------------------
            # Ticks and spines
            # ------------------------------------------------

            ax.yaxis.set_major_locator(
                LogLocator(base=10)
            )

            ax.yaxis.set_major_formatter(
                LogFormatterMathtext(base=10)
            )

            ax.yaxis.set_minor_locator(
                LogLocator(
                    base=10,
                    subs=np.arange(2, 10)
                )
            )

            ax.xaxis.set_minor_locator(
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

            ax.grid(
                show_grid,
                alpha=0.2,
            )

        # ----------------------------------------------------
        # Manual figure layout
        # ----------------------------------------------------

        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
        )

        # ----------------------------------------------------
        # Shared colorbar
        # ----------------------------------------------------

        if show_colorbar:

            # Allocate space immediately to the right
            # of the subplot region
            bar_left = right + colorbar_pad
            bar_right = bar_left + colorbar_fraction

            if bar_right >= 1.0:
                raise ValueError(
                    "Colorbar exceeds figure width. "
                    "Reduce RIGHT, COLORBAR_PAD, or "
                    "COLORBAR_FRACTION."
                )

            cax = fig.add_axes([
                bar_left,
                bottom,
                colorbar_fraction,
                top - bottom,
            ])

            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(
                    norm=norm,
                    cmap=colormap
                ),
                cax=cax,
            )

            colorbar.set_label(
                "Checkpoint",
                fontsize=colorbar_label_size,
            )

            colorbar.ax.tick_params(
                labelsize=tick_size,
                width=major_tick_width,
                length=major_tick_length,
            )

            colorbar.outline.set_linewidth(spine_width)

        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        if save_plot:

            self.output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            suffix = mode

            png = (
                self.output_dir
                / f"{output_name}_{suffix}.png"
            )

            pdf = (
                self.output_dir
                / f"{output_name}_{suffix}.pdf"
            )

            fig.savefig(
                png,
                dpi=dpi,
                bbox_inches="tight",
            )

            fig.savefig(
                pdf,
                bbox_inches="tight",
            )

            print("Saved:", png)
            print("Saved:", pdf)

        # ----------------------------------------------------
        # Display once; do not return figure
        # ----------------------------------------------------

        if show_plot:
            plt.show()

        plt.close(fig)

