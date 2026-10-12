"""Stage PhysNet checkpoint MAE/barrier tables beneath ~/XAI/data and plot them.
No TensorFlow/PhysNet or conda activation required: numpy, pandas, matplotlib.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator

class XAIDataPipeline:
    def __init__(self, xai_root=None, project_dir='project_feature'):
        self.root = Path(xai_root or (Path.home() / 'XAI')).expanduser().resolve()
        self.project = self.root / project_dir
        self.data_dir = self.root / 'data'
        self.sources = {
            'true': {
                'mae': self.project/'models/elec_checkpoint_mae_test.dat',
                'barrier': self.project/'modular_energy/barrier_height/silvan_barrier_elec/barrier_summary_kcal.dat',
            },
            'false': {
                'mae': self.project/'models/t1_checkpoint_mae_test.dat',
                'barrier': self.project/'batch_100/no_elec/barrier_height/no_elec_barrier_optimizations/barrier_summary_fixed.dat',
            },
        }
        self.df = None

    @staticmethod
    def _mae(path):
        rows, columns = [], None
        with open(path) as f:
            for original in f:
                line = original.strip()
                if not line: continue
                if line.startswith('#'):
                    candidate = line.lstrip('# ').split()
                    if 'checkpoint_path' in candidate and 'EMAE_kcalmol' in candidate:
                        columns = candidate[candidate.index('checkpoint_path')+1:]
                    continue
                if columns is None:
                    raise ValueError(f'MAE header not found: {path}')
                parts = line.split()
                positions = [i for i,p in enumerate(parts) if 'model.ckpt' in p]
                if not positions or positions[0] == 0:
                    raise ValueError(f'Checkpoint path not found: {line}')
                i = positions[0]
                if len(parts[i+1:]) != len(columns):
                    raise ValueError(f'MAE header/data mismatch: {line}')
                numbers = dict(zip(columns,parts[i+1:]))
                rows.append({'checkpoint':int(parts[i-1]),
                             'EMAE':float(numbers['EMAE_kcalmol']),
                             'FMAE':float(numbers['FMAE_kcalmolA']),
                             'DMAE':float(numbers['DMAE'])})
        return XAIDataPipeline._validate(rows, path)

    @staticmethod
    def _barrier(path):
        rows = []
        with open(path) as f:
            for original in f:
                line = original.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError(f'Expected 7-column standardized barrier input: {path}: {line}')
                if parts[5] != 'OK' or parts[6] != 'OK': continue
                barrier = float(parts[4])
                if np.isfinite(barrier):
                    rows.append({'checkpoint':int(parts[0]), 'Eb':barrier})
        return XAIDataPipeline._validate(rows,path)

    @staticmethod
    def _validate(rows,path):
        if not rows: raise ValueError(f'No valid rows in {path}')
        df = pd.DataFrame(rows).sort_values('checkpoint')
        if df.checkpoint.duplicated().any(): raise ValueError(f'Duplicate checkpoints in {path}')
        return df

    @staticmethod
    def _write_dat(df, path):
        path.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(path,sep=' ',index=False,na_rep='nan',float_format='%.12g')

    @staticmethod
    def _read_dat(path):
        return pd.read_csv(path,sep=r'\s+',engine='python')

    def extract(self, show_data=False):
        """Read ORIGINAL dat files, export normalized 2/4-column staging files."""
        for mode in ('true','false'):
            for kind, reader in (('MAE',self._mae),('barrier_height',self._barrier)):
                source_key = 'mae' if kind == 'MAE' else 'barrier'
                source = self.sources[mode][source_key]
                if not source.is_file(): raise FileNotFoundError(f'Missing: {source}')
                table = reader(source)
                output = self.data_dir/kind/f'{mode}.dat'
                self._write_dat(table,output)
                print(f'{kind}/{mode}: {len(table)} checkpoints -> {output}')
                if show_data:
                    display(table)
        return self.load(show_data=show_data)

    def load(self, show_data=False, save_merged=True):
        """Load only STAGED files: original simulation directories not needed."""
        tables = []
        for mode in ('true','false'):
            mae = self._read_dat(self.data_dir/'MAE'/f'{mode}.dat')
            barrier = self._read_dat(self.data_dir/'barrier_height'/f'{mode}.dat')
            one = barrier.merge(mae,on='checkpoint',how='outer',validate='one_to_one')
            one = one.rename(columns={k:f'{k}_{mode}' for k in ('Eb','EMAE','FMAE','DMAE')})
            tables.append(one)
        df = tables[0].merge(tables[1],on='checkpoint',how='outer',validate='one_to_one')
        order = ['checkpoint','Eb_true','EMAE_true','FMAE_true','DMAE_true',
                 'Eb_false','EMAE_false','FMAE_false','DMAE_false']
        self.df = df[order].sort_values('checkpoint').reset_index(drop=True)
        if save_merged:
            self._write_dat(self.df,self.data_dir/'combined_9columns.dat')
        if show_data: display(self.df)
        return self.df

    def plot(self, df=None, *, ranges=None, show_plot=True, save_plot=True,
             figsize=(18, 13), colors=("tab:red", "tab:blue"),
             label_size=22, tick_size=18, legend_size=16,
             line_width=3, marker_size=4, spine_width=2.5, dpi=300,
             show_grid=False, show_reference=True,
             reference_checkpoint=1000000, selected_checkpoints=None,
             show_legend=True, major_tick_length=9, minor_tick_length=5,
             major_tick_width=2, minor_tick_width=1.5,
             output_name="barrier_mae_2x2"):
        """Plot four independently zoomable panels; returns no Figure to avoid
        Jupyter's duplicate rendering of the explicitly displayed figure.

        ranges example:
            {"Eb": {"x": (1, 1000), "y": (0, 15)},
             "EMAE": {"x": (100, 1000), "y": (0, 2)}}
        Keys: Eb, EMAE, FMAE, DMAE. None means autoscale.
        """
        df = self.df if df is None else df
        if df is None:
            df = self.load()
        ranges = {} if ranges is None else ranges
        valid_keys = {"Eb", "EMAE", "FMAE", "DMAE"}
        unknown = set(ranges) - valid_keys
        if unknown:
            raise ValueError(f"Unknown panel keys: {sorted(unknown)}")
        panels = [
            ("Eb", r"Barrier height $E_b$ (kcal/mol)"),
            ("EMAE", r"MAE$(E)$ (kcal/mol)"),
            ("FMAE", r"MAE$(F)$ (kcal/mol/Å)"),
            ("DMAE", r"MAE$(\mu)$ (dataset units)"),
        ]
        # sharex=False is essential: each panel must have an independent zoom.
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
        try:
            for ax, (quantity, ylabel) in zip(axes.flat, panels):
                for mode, color in zip(("true", "false"), colors):
                    column = f"{quantity}_{mode}"
                    visible = df[["checkpoint", column]].dropna()
                    visible = visible[visible["checkpoint"] > 0]
                    if selected_checkpoints is not None:
                        visible = visible[
                            visible["checkpoint"].isin(selected_checkpoints)
                        ]
                    visible = visible.sort_values("checkpoint")
                    if not visible.empty:
                        ax.plot(
                            visible["checkpoint"], visible[column],
                            marker="o", markersize=marker_size,
                            linewidth=line_width, color=color,
                            label=f"Electrostatics {mode}",
                        )
                    if quantity == "Eb" and show_reference:
                        ref = df.loc[
                            df["checkpoint"] == reference_checkpoint, column
                        ].dropna()
                        if not ref.empty:
                            ax.axhline(
                                float(ref.iloc[0]), color=color,
                                linestyle="--", linewidth=1.7,
                            )
                ax.set_xscale("log")
                ax.set_ylabel(ylabel, fontsize=label_size)
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_minor_locator(
                    LogLocator(base=10, subs=np.arange(2, 10))
                )
                ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                ax.tick_params(
                    which="major", direction="in", top=True, right=True,
                    labelsize=tick_size, width=major_tick_width,
                    length=major_tick_length,
                )
                ax.tick_params(
                    which="minor", direction="in", top=True, right=True,
                    width=minor_tick_width, length=minor_tick_length,
                )
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                if show_grid:
                    ax.grid(True, alpha=0.2)
                else:
                    ax.grid(False)

                zoom = ranges.get(quantity, {}) or {}
                if set(zoom) - {"x", "y"}:
                    raise ValueError(f"Invalid zoom axis for {quantity}: {zoom}")
                if zoom.get("x") is not None:
                    low, high = zoom["x"]
                    if low <= 0 or high <= low:
                        raise ValueError(
                            f"Log x limits must satisfy 0 < low < high: {zoom['x']}"
                        )
                    ax.set_xlim(low, high)
                if zoom.get("y") is not None:
                    ax.set_ylim(*zoom["y"])

            for ax in axes[1]:
                ax.set_xlabel("Transfer-learning steps", fontsize=label_size)
            if show_legend:
                axes[0, 0].legend(frameon=False, fontsize=legend_size)
            fig.tight_layout()
            if save_plot:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.data_dir / f"{output_name}.png", dpi=dpi,
                            bbox_inches="tight")
                fig.savefig(self.data_dir / f"{output_name}.pdf",
                            bbox_inches="tight")
            if show_plot:
                plt.show()
        finally:
            plt.close(fig)
        # Avoid returning fig: an unassigned figure at the end of a notebook
        # cell would otherwise be displayed again by Jupyter.
        return None

    def run(self, *, extract=False, show_data=True, save_merged=True,
            show_plot=True, **plot_options):
        """Single entry point from Jupyter: stage/load, display, plot."""
        if extract:
            df = self.extract(show_data=False)
        else:
            df = self.load(show_data=False, save_merged=save_merged)
        if show_data:
            from IPython.display import display
            display(df)
        if show_plot or plot_options.get("save_plot", True):
            self.plot(df, show_plot=show_plot, **plot_options)
        return df
