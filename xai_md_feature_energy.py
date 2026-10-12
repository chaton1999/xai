#!/usr/bin/env python3
"""Stage and plot PhysNet feature magnitude vs atomic module energy along MD.

Source: md_from_silvan/[KC_md_pt_featurechange/]activations/
Output: ~/XAI/data/md_feature_energy/
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

EV_TO_KCAL = 23.060548

class MDFeatureEnergy:
    def __init__(self, xai_root=None, source_dir=None):
        self.root = Path(xai_root or Path.home() / 'XAI').expanduser()
        self.output = self.root / 'data' / 'md_feature_energy'
        if source_dir is None:
            base = self.root / 'project_feature' / 'md_from_silvan'
            candidates = [
                base / 'activations',
                base / 'KC_md_pt_featurechange' / 'activations',
                base,
            ]
            for candidate in candidates:
                if (candidate / 'act_hlpes_700K_md_opt_ma_a.xyz.traj.npz').is_file():
                    source_dir = candidate
                    break
            if source_dir is None:
                raise FileNotFoundError('Could not find activation NPZ in:\n' + '\n'.join(map(str, candidates)) + '\nPass source_dir=... to MDFeatureEnergy().')
        self.source = Path(source_dir).expanduser()
        self.stage_path = self.output / 'feature_energy_700K_frames_3700_4019.npz'

    def extract(self, first_frame=3700, last_frame=4020, energy_unit='eV',
                act_filename='act_hlpes_700K_md_opt_ma_a.xyz.traj.npz',
                energy_filename='atomic_energies_700K.npz',
                stage_name=None):
        """first_frame inclusive, last_frame exclusive, both zero-based.

        npz['act'] has [frame, embedding+5 modules, atom, feature].
        Ea_mod either contains the selected window or entire trajectory.
        """
        if first_frame < 0 or last_frame <= first_frame:
            raise ValueError('Invalid frame window')
        act_file = self.source / act_filename
        energy_file = self.source / energy_filename
        print('Source:', self.source, flush=True)
        print('Loading activation array (source NPZ may be ~1.1 GB)...', flush=True)
        with np.load(act_file, allow_pickle=False) as npz:
            full = npz['act']
            if full.ndim != 4 or full.shape[1] < 6 or full.shape[0] < last_frame:
                raise ValueError(f'Unexpected activation shape {full.shape}; expected [frames, >=6, atoms, features].')
            # Avoid persisting the full GB array in the staged file.
            magnitude = np.linalg.norm(full[first_frame:last_frame, 1:6, :, :], axis=-1)
            del full
        print('Loading atomic energy array...', flush=True)
        with np.load(energy_file, allow_pickle=False) as npz:
            energy = np.asarray(npz['Ea_mod'], dtype=float)
        n = last_frame - first_frame
        if energy.ndim != 3 or energy.shape[1] < 5:
            raise ValueError(f'Expected Ea_mod [frames,5,atoms]; found {energy.shape}')
        if energy.shape[0] == n:
            selected_energy = energy[:, :5, :]
        elif energy.shape[0] >= last_frame:
            selected_energy = energy[first_frame:last_frame, :5, :]
        else:
            raise ValueError(f'Ea_mod has {energy.shape[0]} frames but requires {n} or at least {last_frame}.')
        if selected_energy.shape != magnitude.shape:
            raise ValueError(f'Magnitude {magnitude.shape} != energy {selected_energy.shape}')
        if energy_unit == 'eV':
            selected_energy = selected_energy * EV_TO_KCAL
        elif energy_unit != 'kcal/mol':
            raise ValueError("energy_unit must be 'eV' or 'kcal/mol'")
        self.output.mkdir(parents=True, exist_ok=True)
        output = self.output / (stage_name or f'feature_energy_700K_frames_{first_frame}_{last_frame-1}.npz')
        np.savez_compressed(output,
                            frames=np.arange(first_frame, last_frame),
                            magnitude=magnitude, energy_kcalmol=selected_energy)
        self.stage_path = output
        print('Staged:', output, '| arrays:', magnitude.shape, selected_energy.shape, flush=True)
        return output

    def load(self, stage_path=None):
        path = Path(stage_path) if stage_path is not None else self.stage_path
        if not path.is_file():
            raise FileNotFoundError(f'Staged data not found: {path}; run extract() once.')
        with np.load(path, allow_pickle=False) as npz:
            return {key: npz[key] for key in ('frames', 'magnitude', 'energy_kcalmol')}

    def plot(self, data=None, modules=(1,2,3,4,5),
             atoms=((1,'C1'),(7,'O7'),(8,'H8')),
             figsize=(12.5,15), dpi=300,
             point_color='#67cfff', point_size=12, point_alpha=.65,
             start_color='orange', end_color='red', endpoint_size=46,
             xlabel=r'$\|\mathcal{F}^{\alpha}\|$',
             ylabel=r'$E_a$ (kcal/mol)',
             label_size=16, tick_size=12, module_label_size=20,
             column_label_size=15, spine_width=1.1,
             major_tick_length=5, major_tick_width=1.2,
             minor_tick_length=2.5, minor_tick_width=.8, minor_subdivisions=2,
             left=.09, right=.89, bottom=.075, top=.985,
             wspace=.36, hspace=.45,
             xlims=None, ylims=None, show_grid=False,
             show_column_titles=False, show=True, save=True,
             output_name='figure_feature_magnitude_vs_atomic_energy_5x3'):
        """atoms are (1-based index, display name). Per-panel limits keyed (module, atom)."""
        if data is None:
            data = self.load()
        mag, energy = data['magnitude'], data['energy_kcalmol']
        xlims = xlims or {}; ylims = ylims or {}
        fig, axes = plt.subplots(len(modules), len(atoms), figsize=figsize, squeeze=False)
        for i, module in enumerate(modules):
            if module < 1 or module > mag.shape[1]:
                raise ValueError(f'Module {module} unavailable.')
            for j, (atom, name) in enumerate(atoms):
                a = atom - 1
                if a < 0 or a >= mag.shape[2]:
                    raise ValueError(f'Atom {atom} unavailable.')
                ax = axes[i,j]
                x = mag[:,module-1,a]
                y = energy[:,module-1,a]
                ax.scatter(x,y,s=point_size,c=point_color,alpha=point_alpha,edgecolors='none',rasterized=True)
                ax.scatter([x[0]],[y[0]],s=endpoint_size,c=start_color,edgecolors='none',zorder=5)
                ax.scatter([x[-1]],[y[-1]],s=endpoint_size,c=end_color,edgecolors='none',zorder=6)
                if (module,atom) in xlims: ax.set_xlim(*xlims[(module,atom)])
                if (module,atom) in ylims: ax.set_ylim(*ylims[(module,atom)])
                for spine in ax.spines.values(): spine.set_linewidth(spine_width)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(which='major',length=major_tick_length,width=major_tick_width,labelsize=tick_size)
                ax.tick_params(which='minor',length=minor_tick_length,width=minor_tick_width)
                if minor_subdivisions > 0:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_subdivisions))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_subdivisions))
                if j == 0: ax.set_ylabel(ylabel,fontsize=label_size)
                if i == len(modules)-1: ax.set_xlabel(xlabel,fontsize=label_size)
                if show_column_titles and i == 0: ax.set_title(name,fontsize=column_label_size)
                ax.grid(show_grid,alpha=.25)
            # Annotate rows in figure coordinates after adjustment below.
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
        for i, module in enumerate(modules):
            bbox = axes[i,-1].get_position()
            fig.text(min(0.995,right + .025), .5*(bbox.y0+bbox.y1), f'M{module}',
                     ha='left',va='center',fontsize=module_label_size)
        if save:
            self.output.mkdir(parents=True,exist_ok=True)
            for suffix in ('png','pdf'):
                path=self.output/f'{output_name}.{suffix}'
                fig.savefig(path,dpi=dpi,bbox_inches='tight')
                print('Saved:',path)
        if show: plt.show()
        plt.close(fig)
        return axes

