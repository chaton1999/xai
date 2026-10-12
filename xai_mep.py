#!/usr/bin/env python3
"""Extract MEP NPZ results once, then plot staged pandas tables.

Default source: ~/XAI/project_feature/modular_energy/mep_yes/1000000/npz_1m
Destination:    ~/XAI/MEP/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

class MEPAnalysis:
    ATOMS = {'C1':5, 'C2':4, 'H3':7, 'C4':3, 'H5':6,
             'H6':8, 'O7':0, 'H8':2, 'O9':1}
    ELEMENT_Z = {'C':6, 'H':1, 'O':8}
    BONDS = [('C1','H3'), ('C2','H5'), ('C4','H6'),
             ('O7','H8'), ('O9','H8'), ('C4','O9'),
             ('C1','O7'), ('C4','C2'), ('C1','C2')]
    BOND_COLORS = {'C1-H3':'cyan','C2-H5':'blue','C4-H6':'slateblue',
                   'O7-H8':'tab:green','O9-H8':'limegreen',
                   'C4-O9':'green','C1-O7':'lime',
                   'C4-C2':'darkred','C1-C2':'red'}
    EV_TO_KCAL = 23.060548

    def __init__(self, xai_root=None, source=None, output=None):
        self.root = Path(xai_root or Path.home()/'XAI').expanduser().resolve()
        self.source = Path(source).expanduser().resolve() if source else (
            self.root/'project_feature/modular_energy/mep_yes/1000000/npz_1m')
        self.output = Path(output).expanduser().resolve() if output else self.root/'MEP'
        self.tables = {}

    @staticmethod
    def _atom_vector(a, key, n=9):
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size != n:
            raise ValueError(f'{key} expected {n} atoms; got shape {a.shape}')
        return a

    def extract(self, start=1, end=600, progress_every=25, feature_key='act',
                overwrite=True):
        """Read each NPZ once; write bonds.dat, atoms.dat, molecular.dat.

        Frames missing from range cause an error rather than silently changing the MEP.
        """
        if start < 1 or end < start:
            raise ValueError('Require 1 <= start <= end')
        self.output.mkdir(parents=True, exist_ok=True)
        outputs = [self.output/f for f in ('bonds.dat','atoms.dat','molecular.dat')]
        if not overwrite and any(p.exists() for p in outputs):
            raise FileExistsError('Staged tables exist; set overwrite=True or load instead')
        atom_rows, mol_rows, bond_rows = [], [], []
        z_reference = None
        reference_bonds = None
        import time
        tic = time.perf_counter()
        count = end-start+1
        for i, frame in enumerate(range(start, end+1), 1):
            filename = self.source/f'mp2_perblock_analysis_{frame}.npz'
            if not filename.is_file():
                raise FileNotFoundError(f'Missing MEP frame {frame}: {filename}')
            with np.load(filename, allow_pickle=True) as d:
                R = np.asarray(d['R'], dtype=float)
                Z = np.asarray(d['Z'], dtype=int).reshape(-1)
                if R.shape != (9,3) or Z.shape != (9,):
                    raise ValueError(f'Unexpected R/Z shapes in {filename}: {R.shape}/{Z.shape}')
                if z_reference is None:
                    z_reference = Z.copy()
                    for label, idx in self.ATOMS.items():
                        if Z[idx] != self.ELEMENT_Z[label[0]]:
                            raise ValueError(f'Atom mapping failed for {label} at index {idx}')
                elif not np.array_equal(Z, z_reference):
                    raise ValueError(f'Atom identity changed at frame {frame}')

                if 'E_blocks_scaled_no_shift' in d.files and 'E_shift_atom' in d.files:
                    blocks = np.asarray(d['E_blocks_scaled_no_shift'], dtype=float)
                    if blocks.ndim != 2 or blocks.shape[1] != 9:
                        raise ValueError(f'Invalid block energies in {filename}')
                    Enn = blocks.sum(axis=0) + self._atom_vector(d['E_shift_atom'], 'E_shift_atom')
                elif 'E_atom' in d.files:
                    Enn = self._atom_vector(d['E_atom'], 'E_atom')
                else:
                    raise KeyError(f'No per-atom neural-network energy in {filename}')

                elec_key = next((k for k in ('E_elec_atom','E_elec') if k in d.files), None)
                if elec_key is None:
                    raise KeyError(f'No per-atom electrostatic energy in {filename}')
                Eelec = np.asarray(d[elec_key], dtype=float)
                if Eelec.shape == (9,):
                    pass
                elif Eelec.ndim == 2 and Eelec.shape[1] == 9:
                    Eelec = Eelec[-1]
                elif Eelec.ndim == 2 and Eelec.shape[0] == 9:
                    Eelec = Eelec[:,-1]
                else:
                    raise ValueError(f'Unexpected {elec_key} shape {Eelec.shape} in {filename}')

                if feature_key not in d.files:
                    raise KeyError(f'Missing {feature_key!r} in {filename}; available: {d.files}')
                act = np.asarray(d[feature_key], dtype=float)
                if act.ndim == 3:
                    last = act[-1]
                elif act.ndim == 2:
                    last = act
                else:
                    raise ValueError(f'Unexpected {feature_key} shape {act.shape} in {filename}')
                if last.shape[0] != 9 and last.shape[1] == 9:
                    last = last.T
                if last.ndim != 2 or last.shape[0] != 9:
                    raise ValueError(f'Unexpected final feature shape {last.shape} in {filename}')
                mag = np.linalg.norm(last, axis=1)

            Etot = Enn+Eelec
            atom_row = {'frame':frame}
            for label, idx in self.ATOMS.items():
                atom_row[f'ENN_{label}'] = Enn[idx]
                atom_row[f'Eelec_{label}'] = Eelec[idx]
                atom_row[f'Etot_{label}'] = Etot[idx]
                atom_row[f'mag_{label}'] = mag[idx]
            atom_rows.append(atom_row)
            mol_rows.append({'frame':frame,'ENN_eV':Enn.sum(),
                             'Eelec_eV':Eelec.sum(),'Etot_eV':Etot.sum(),
                             'mag':np.linalg.norm(last.reshape(-1))})
            distances = {f'{a}-{b}':np.linalg.norm(R[self.ATOMS[a]]-R[self.ATOMS[b]])
                         for a,b in self.BONDS}
            if reference_bonds is None:
                reference_bonds = distances.copy()
            bond_row = {'frame':frame}
            for bond, dist in distances.items():
                code = bond.replace('-','_')
                bond_row[f'r_{code}'] = dist
                bond_row[f'dr_{code}'] = dist-reference_bonds[bond]
            bond_rows.append(bond_row)
            if progress_every and (i == 1 or i % progress_every == 0 or i == count):
                elapsed = max(time.perf_counter()-tic, 1e-9)
                rate = i/elapsed
                print(f'Extract {i}/{count} ({100*i/count:5.1f}%) | '
                      f'{rate:.1f} frames/s | ETA {(count-i)/rate:.1f}s', flush=True)
        self.tables = {'bonds':pd.DataFrame(bond_rows),
                       'atoms':pd.DataFrame(atom_rows),
                       'molecular':pd.DataFrame(mol_rows)}
        for key, df in self.tables.items():
            path = self.output/f'{key}.dat'
            df.to_csv(path, sep=' ', index=False, float_format='%.12g')
            print(f'Saved {path} ({len(df)} rows)')
        return self.tables

    def load(self):
        self.tables = {key:pd.read_csv(self.output/f'{key}.dat', sep=r'\s+')
                       for key in ('bonds','atoms','molecular')}
        return self.tables

    def _get(self, key):
        if key not in self.tables:
            self.load()
        return self.tables[key]

    @staticmethod
    def _style(ax, label_size, tick_size, spine_width, major_length, minor_length):
        ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,
                       labelsize=tick_size,width=spine_width,length=major_length)
        ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,
                       width=spine_width*.7,length=minor_length)
        for s in ax.spines.values(): s.set_linewidth(spine_width)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    def _finish(self, fig, name, show, save, dpi):
        if save:
            self.output.mkdir(parents=True, exist_ok=True)
            for ext in ('png','pdf'):
                fig.savefig(self.output/f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
            print('Saved figure:', self.output/name)
        if show: plt.show()
        plt.close(fig)

    def plot_bonds(self, frame_range=(1,300), bonds=None, delta=True,
                   ylim=None, figsize=(12,7), show=True, save=True,
                   legend=True, show_grid=False, linewidth=3,
                   label_size=20, tick_size=17, spine_width=2.5,
                   major_length=9, minor_length=5, dpi=300,
                   left=.13,right=.97,bottom=.16,top=.94,
                   output_name='bond_time_series'):
        df = self._get('bonds')
        shown = df[df.frame.between(*frame_range)]
        if shown.empty: raise ValueError('No bond frames in selected range')
        bonds = bonds or [f'{a}-{b}' for a,b in self.BONDS]
        fig,ax=plt.subplots(figsize=figsize)
        for bond in bonds:
            col=('dr_' if delta else 'r_')+bond.replace('-','_')
            ax.plot(shown.frame,shown[col],lw=linewidth,
                    color=self.BOND_COLORS.get(bond),label=bond)
        if delta: ax.axhline(0,color='black',lw=1,ls='--',alpha=.5)
        if frame_range[0] <= 300 <= frame_range[1]:
            ax.axvline(300,color='black',ls='--',lw=1.5,alpha=.6)
        ax.set_xlabel('Geometry index',fontsize=label_size)
        ax.set_ylabel(r'$\Delta r$ ($\AA$)' if delta else r'$r$ ($\AA$)',fontsize=label_size)
        ax.set_xlim(*frame_range)
        if ylim is not None: ax.set_ylim(*ylim)
        self._style(ax,label_size,tick_size,spine_width,major_length,minor_length)
        ax.grid(show_grid,alpha=.15)
        if legend: ax.legend(frameon=False,fontsize=max(10,tick_size-3),ncol=3)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
        self._finish(fig,output_name,show,save,dpi)

    def plot_atoms(self, frame_range=(1,300), atoms=None, energy_mode='total',
                   xlimits=None, ylimits=None, figsize=(17,13), stride=1,
                   cmap='Blues_r', marker_size=65, ts_marker_size=130,
                   show_colorbar=True, show=True, save=True, show_grid=False,
                   label_size=20,tick_size=15,title_size=18,spine_width=2.3,
                   major_length=8,minor_length=4,dpi=300,
                   left=.09,right=.86,bottom=.09,top=.94,wspace=.35,hspace=.36,
                   output_name='atom_energy_vs_magnitude'):
        if energy_mode not in ('total','nn','elec'):
            raise ValueError('energy_mode must be total, nn, or elec')
        df=self._get('atoms')
        shown=df[df.frame.between(*frame_range)].iloc[::stride]
        if shown.empty: raise ValueError('No atom frames in selected range')
        atoms=list(atoms or self.ATOMS.keys())
        ncol=3
        nrow=int(np.ceil(len(atoms)/ncol))
        fig,axes=plt.subplots(nrow,ncol,figsize=figsize,squeeze=False)
        sm=plt.cm.ScalarMappable(norm=Normalize(vmin=frame_range[0],vmax=frame_range[1]),cmap=cmap)
        for ax,atom in zip(axes.flat,atoms):
            key={'total':'Etot','nn':'ENN','elec':'Eelec'}[energy_mode]
            x=shown[f'{key}_{atom}'].to_numpy()*self.EV_TO_KCAL
            y=shown[f'mag_{atom}'].to_numpy()
            frames=shown.frame.to_numpy()
            pathmask=(frames!=300)
            ax.scatter(x[pathmask],y[pathmask],c=frames[pathmask],cmap=cmap,
                       norm=sm.norm,s=marker_size,linewidths=0,alpha=.85)
            tsmask=(frames==300)
            if tsmask.any():
                ax.scatter(x[tsmask],y[tsmask],s=ts_marker_size,
                           c='black',marker='X',zorder=10)
            if isinstance(xlimits,dict) and atom in xlimits: ax.set_xlim(*xlimits[atom])
            elif xlimits is not None and not isinstance(xlimits,dict): ax.set_xlim(*xlimits)
            if isinstance(ylimits,dict) and atom in ylimits: ax.set_ylim(*ylimits[atom])
            elif ylimits is not None and not isinstance(ylimits,dict): ax.set_ylim(*ylimits)
            ax.set_title(atom,fontsize=title_size)
            ax.set_xlabel('Atomic energy (kcal/mol)',fontsize=label_size)
            ax.set_ylabel(r'$\|\mathcal{F}_i^{(5)}\|$',fontsize=label_size)
            self._style(ax,label_size,tick_size,spine_width,major_length,minor_length)
            ax.grid(show_grid,alpha=.15)
        for ax in axes.flat[len(atoms):]: ax.set_visible(False)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,
                            wspace=wspace,hspace=hspace)
        if show_colorbar:
            bar=fig.add_axes([right+.025,bottom,.022,top-bottom])
            cb=fig.colorbar(sm,cax=bar)
            cb.set_label('Geometry index',fontsize=label_size)
            cb.ax.tick_params(labelsize=tick_size)
        self._finish(fig,output_name,show,save,dpi)

    def plot_molecular(self, frame_range=(1,600), stride=5, relative_energy=True,
                       energy_mode='total', ylim_mag=None, ylim_energy=None,
                       figsize=(16,8), mag_color='tab:blue',energy_color='tab:red',
                       mag_marker='o',energy_marker='s',marker_size=45,
                       show=True,save=True,show_grid=False,show_legend=True,
                       label_size=22,tick_size=18,spine_width=2.5,
                       major_length=9,minor_length=5,dpi=300,
                       left=.13,right=.85,bottom=.15,top=.93,
                       output_name='molecular_energy_magnitude'):
        if energy_mode not in ('total','nn','elec'):
            raise ValueError('energy_mode must be total, nn, or elec')
        df=self._get('molecular')
        complete=df[df.frame.between(*frame_range)]
        if complete.empty: raise ValueError('No molecular frames in selected range')
        shown=complete.iloc[::stride]
        e_col={'total':'Etot_eV','nn':'ENN_eV','elec':'Eelec_eV'}[energy_mode]
        e0=complete[e_col].min() if relative_energy else 0.0
        energies=(shown[e_col]-e0)*self.EV_TO_KCAL
        fig,ax=plt.subplots(figsize=figsize)
        ax2=ax.twinx()
        h1=ax.scatter(shown.frame,shown.mag,c=mag_color,s=marker_size,
                      marker=mag_marker,label='Molecular feature magnitude')
        h2=ax2.scatter(shown.frame,energies,c=energy_color,s=marker_size,
                       marker=energy_marker,label='Molecular energy')
        ax.set_xlim(*frame_range)
        if ylim_mag is not None: ax.set_ylim(*ylim_mag)
        if ylim_energy is not None: ax2.set_ylim(*ylim_energy)
        ax.set_xlabel('Geometry index',fontsize=label_size)
        ax.set_ylabel(r'$\|\mathcal{F}_{\mathrm{mol}}^{(5)}\|$',fontsize=label_size)
        ax2.set_ylabel(r'$E-E_{\min}$ (kcal/mol)' if relative_energy else
                       r'$E$ (kcal/mol)',fontsize=label_size)
        self._style(ax,label_size,tick_size,spine_width,major_length,minor_length)
        ax2.tick_params(axis='y',labelsize=tick_size,width=spine_width,length=major_length)
        for s in ax2.spines.values(): s.set_linewidth(spine_width)
        ax.grid(show_grid,alpha=.15)
        if show_legend: ax.legend([h1,h2],['Molecular feature magnitude','Molecular energy'],
                                  fontsize=max(10,tick_size-2),frameon=False,loc='best')
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
        self._finish(fig,output_name,show,save,dpi)

    # ============================================================
    # Wiberg bond order + molecular structure snapshots
    # ============================================================
    STRUCTURE_BONDS = ('C4-O9','C1-O7','C4-C2','C1-C2')
    STRUCTURE_COLORS = {'C4-O9':'tab:green','C1-O7':'tab:green',
                        'C4-C2':'tab:red','C1-C2':'tab:red'}
    STRUCTURE_MARKERS = {'C4-O9':'D','C1-O7':'*','C4-C2':'s','C1-C2':'o'}

    def extract_structures(self, xyz_file=None, wiberg_file=None,
                           frame_offset=1, progress_every=25):
        """Stage 100-frame XYZ and Wiberg data to XAI/MEP.

        The original Wiberg table uses zero-based frame indices, so by default
        frame_offset=1. There must be a valid Wiberg row for each snapshot
        requested later; incomplete Wiberg series are allowed at extraction.
        """
        xyz_file = Path(xyz_file) if xyz_file is not None else (
            self.source/'MEP'/'100f_latest_combined.xyz')
        wiberg_file = Path(wiberg_file) if wiberg_file is not None else (
            self.source/'MEP'/'wiberg_bond_order_reindexed.dat')
        if not xyz_file.is_file():
            raise FileNotFoundError(f'Cannot find XYZ trajectory: {xyz_file}')
        if not wiberg_file.is_file():
            raise FileNotFoundError(f'Cannot find Wiberg table: {wiberg_file}')

        coords = []
        expected = ['O','O','H','C','C','C','H','H','H']
        with xyz_file.open() as handle:
            while True:
                line = handle.readline()
                if not line: break
                if not line.strip(): continue
                n = int(line.strip())
                if n != 9: raise ValueError(f'Expected 9 atoms in {xyz_file}, got {n}')
                comment = handle.readline()
                if not comment: raise ValueError('Truncated XYZ comment line')
                symbols, positions = [], []
                for _ in range(n):
                    parts = handle.readline().split()
                    if len(parts) < 4: raise ValueError('Truncated XYZ atom line')
                    symbols.append(parts[0])
                    positions.append([float(v) for v in parts[1:4]])
                if symbols != expected:
                    raise ValueError(f'Unexpected XYZ atom order at geometry {len(coords)+1}: {symbols}')
                coords.append(positions)
                if progress_every and len(coords) % progress_every == 0:
                    print(f'Read {len(coords)} XYZ geometries', flush=True)
        if not coords: raise ValueError(f'No geometries in {xyz_file}')
        coords = np.asarray(coords, dtype=float)
        if not np.isfinite(coords).all(): raise ValueError('Nonfinite XYZ coordinates')

        table = pd.read_csv(wiberg_file, sep=r'\s+', comment='#')
        if 'frame' not in table:
            raise ValueError(f'Wiberg file requires a frame column: {wiberg_file}')
        table['frame'] = pd.to_numeric(table['frame'], errors='raise').astype(int) + frame_offset
        if table['frame'].duplicated().any():
            raise ValueError('Duplicate Wiberg frame indices')
        result = pd.DataFrame({'frame':table['frame']})
        for pair in self.STRUCTURE_BONDS:
            a,b = pair.split('-')
            name = f'{a}_{b}' if f'{a}_{b}' in table else f'{b}_{a}'
            if name not in table:
                raise ValueError(f'Wiberg column {a}_{b} / {b}_{a} missing')
            result[pair] = pd.to_numeric(table[name], errors='raise')
        if not np.isfinite(result[list(self.STRUCTURE_BONDS)].to_numpy()).all():
            raise ValueError('Nonfinite Wiberg bond indices')
        self.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output/'structures_100.npz', R=coords,
                            original_symbols=np.array(expected))
        result = result.sort_values('frame')
        result.to_csv(self.output/'wiberg_structures.dat', sep=' ', index=False,
                      float_format='%.10g')
        print(f'Staged {len(coords)} structures and {len(result)} Wiberg frames in {self.output}')
        return coords, result

    @staticmethod
    def _project_snapshots(coords, atom_order, align=True):
        """Project a series of geometries onto the reference molecular plane."""
        geometries = coords[:,atom_order,:]
        heavy = [0,1,3,6,8]  # C1,C2,C4,O7,O9
        reference = geometries[0] - geometries[0,heavy].mean(axis=0)
        aligned = []
        for geometry in geometries:
            centered = geometry - geometry[heavy].mean(axis=0)
            if align:
                U,_,Vt = np.linalg.svd(centered[heavy].T @ reference[heavy])
                correction = np.diag([1.,1.,np.linalg.det(U @ Vt)])
                centered = centered @ (U @ correction @ Vt)
            aligned.append(centered)
        _,_,vt = np.linalg.svd(reference[heavy], full_matrices=False)
        normal = vt[-1]
        horizontal = reference[3] - reference[0]
        horizontal -= np.dot(horizontal,normal)*normal
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal,horizontal)
        if np.mean(reference[[6,8]] @ vertical) < 0: vertical = -vertical
        return np.asarray(aligned) @ np.column_stack([horizontal,vertical])

    @staticmethod
    def _choose_frames(mode, n_total, count=10, ts_reduced=50):
        """10 on RS→TS, or 5 on each branch for a full path."""
        if mode not in ('rs_ts','full'):
            raise ValueError('path_mode must be "rs_ts" or "full"')
        if mode == 'rs_ts':
            max_frame = min(n_total, ts_reduced)
            if max_frame < count:
                raise ValueError('Not enough frames for requested snapshots')
            frames = np.rint(np.linspace(1,max_frame,count)).astype(int)
        else:
            if count % 2: raise ValueError('Full-path snapshot count must be even (e.g. 10)')
            half = count//2
            if not 1 < ts_reduced < n_total:
                raise ValueError('TS reduced index must lie inside full path')
            first = np.rint(np.linspace(1,ts_reduced,half)).astype(int)
            second = np.rint(np.linspace(ts_reduced+1,n_total,half)).astype(int)
            frames = np.concatenate([first,second])
        if len(set(frames)) != len(frames):
            raise ValueError('Snapshot selection produced duplicates; reduce count')
        return frames

    def plot_wiberg_structures(
        self, path_mode='rs_ts', n_structures=10, ts_reduced_frame=50,
        annotate='original', frame_multiplier=6, frame_offset=0,
        snapshot_frames=None, show=True, save=True, figsize=(16,11),
        plot_changes=False, show_snapshot_lines=False,
        show_bo_numbers=False, show_hydrogen_bonds=True,
        align_geometries=True, bo_min=.8, bo_max=1.6,
        width_min=1., width_max=8., atom_label_size=12,
        snapshot_title_size=15, label_size=22, tick_size=18,
        legend_size=15, legend_ncol=2, marker_size=45,
        bond_alpha=1., alpha=.95, panel_label_size=24,
        spine_width=2., major_length=8, minor_length=4,
        wspace=.10, hspace=.35, upper_lower_hspace=.28,
        left=.09, right=.97, bottom=.07, top=.92,
        top_height=1.3, bottom_height=1.85,
        output_name='wiberg_structures', dpi=300):
        """Plot Wiberg-vs-reduced-frame and 10 snapshot structures.

        Default 100-point full MEP has TS at reduced frame 50. Original
        figure's annotations were 6*reduced_frame (6,12,...,300,...,600).
        Change frame_multiplier / frame_offset if your sampling differs.
        """
        if bo_min >= bo_max:
            raise ValueError('bo_min must be smaller than bo_max')
        cache = self.output/'structures_100.npz'
        bo_file = self.output/'wiberg_structures.dat'
        if not cache.exists() or not bo_file.exists():
            raise FileNotFoundError('Run extract_structures() first to stage XYZ and Wiberg files')
        with np.load(cache, allow_pickle=False) as d:
            geometries = np.asarray(d['R'], dtype=float)
        bo = pd.read_csv(bo_file, sep=r'\s+')
        total = len(geometries)
        end = ts_reduced_frame if path_mode == 'rs_ts' else total
        if not 1 < ts_reduced_frame <= total:
            raise ValueError('TS reduced frame outside available geometries')
        if snapshot_frames is None:
            selected = self._choose_frames(path_mode,total,n_structures,ts_reduced_frame)
        else:
            selected = np.asarray(snapshot_frames,dtype=int)
            if np.any((selected < 1)|(selected > end)):
                raise ValueError('Snapshot frames outside selected path')
        # A structure can still be displayed without a matching Wiberg row.
        # Such bonds are drawn gray rather than inventing bond-order values.
        have = set(bo.frame.astype(int))
        missing = [int(f) for f in selected if f not in have]
        if missing:
            print(f'Warning: no Wiberg data at reduced frames {missing}; '
                  'draw corresponding colored bonds in gray (unknown BO).')
        bo = bo.set_index('frame').sort_index()
        indices = np.array([self.ATOMS[k] for k in self.ATOMS])
        labels = list(self.ATOMS)
        xy = self._project_snapshots(geometries,indices,align_geometries)
        chosen_xy = xy[selected-1]
        lim_lo=chosen_xy.min(axis=(0,1))-.45
        lim_hi=chosen_xy.max(axis=(0,1))+.45
        fig=plt.figure(figsize=figsize)
        main=fig.add_gridspec(2,1,height_ratios=[top_height,bottom_height],hspace=upper_lower_hspace)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
        ax=fig.add_subplot(main[0])
        for pair in self.STRUCTURE_BONDS:
            values=bo[pair]
            values=values[(values.index>=1)&(values.index<=end)]
            if plot_changes and not values.empty:
                values=values-values.iloc[0]
            ax.scatter(values.index,values.values,marker=self.STRUCTURE_MARKERS[pair],
                       color=self.STRUCTURE_COLORS[pair],s=marker_size,
                       alpha=alpha,label=pair,edgecolors='none')
        if show_snapshot_lines:
            for frame in selected:
                ax.axvline(frame,c='0.85',lw=.7,ls=':',zorder=0)
        ax.axvline(ts_reduced_frame,color='0.3',lw=1.1,ls='--',alpha=.6)
        ax.set_xlim(1,end)
        ax.set_xticks([1,ts_reduced_frame] if path_mode=='rs_ts' else [1,ts_reduced_frame,total])
        ax.set_xticklabels(['RS','TS'] if path_mode=='rs_ts' else ['RS','TS','PS'])
        ax.set_ylabel(r'$\Delta W$' if plot_changes else 'Wiberg bond index',fontsize=label_size)
        self._style(ax,label_size,tick_size,spine_width,major_length,minor_length)
        ax.legend(ncol=legend_ncol,frameon=False,fontsize=legend_size,loc='best')
        ax.text(-.05,1.04,'A',transform=ax.transAxes,fontsize=panel_label_size,fontweight='bold')
        cols=5
        rows=int(np.ceil(len(selected)/cols))
        bottom_grid=main[1].subgridspec(rows,cols,wspace=wspace,hspace=hspace)
        idx={label:i for i,label in enumerate(labels)}
        for j,frame in enumerate(selected):
            sub=fig.add_subplot(bottom_grid[j//cols,j%cols])
            pos=xy[frame-1]
            xyz=geometries[frame-1][indices]
            row=bo.loc[int(frame)] if int(frame) in bo.index else None
            for pair in self.STRUCTURE_BONDS:
                a,b=pair.split('-')
                value=float(row[pair]) if row is not None else np.nan
                strength=np.clip((value-bo_min)/(bo_max-bo_min),0,1) if row is not None else 0
                lw=width_min+(width_max-width_min)*strength
                sub.plot([pos[idx[a],0],pos[idx[b],0]],
                         [pos[idx[a],1],pos[idx[b],1]],
                         color=self.STRUCTURE_COLORS[pair] if row is not None else '.65',lw=lw,alpha=bond_alpha,
                         solid_capstyle='round',zorder=2)
                if show_bo_numbers and row is not None:
                    mid=(pos[idx[a]]+pos[idx[b]])/2
                    sub.text(*mid,f'{value:.2f}',fontsize=9,ha='center',va='center',
                             bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=6)
            for a,b in (('C1','H3'),('C2','H5'),('C4','H6')):
                sub.plot([pos[idx[a],0],pos[idx[b],0]],
                         [pos[idx[a],1],pos[idx[b],1]],c='.55',lw=1.1,zorder=1)
            if show_hydrogen_bonds:
                oxygen=('O7','O9')
                distances=[np.linalg.norm(xyz[idx[o]]-xyz[idx['H8']]) for o in oxygen]
                nearest=int(np.argmin(distances))
                for k,o in enumerate(oxygen):
                    sub.plot([pos[idx[o],0],pos[idx['H8'],0]],
                             [pos[idx[o],1],pos[idx['H8'],1]],c='.55',lw=1.1,
                             ls='-' if k==nearest else '--',zorder=1)
            for name,xy0 in zip(labels,pos):
                sub.text(*xy0,name,ha='center',va='center',fontsize=atom_label_size,zorder=5,
                         bbox=dict(boxstyle='round,pad=0.10',fc='white',ec='none'))
            sub.set_xlim(lim_lo[0],lim_hi[0])
            sub.set_ylim(lim_lo[1],lim_hi[1])
            sub.set_aspect('equal')
            sub.axis('off')
            if frame==1: title='RS'
            elif frame==ts_reduced_frame: title='TS'
            elif frame==total and path_mode=='full': title='PS'
            elif annotate=='original': title=f'Frame {frame_multiplier*frame+frame_offset}'
            elif annotate=='reduced': title=f'Frame {frame}'
            else: title=''
            sub.set_title(title,fontsize=snapshot_title_size)
            if j==0:
                sub.text(-.1,1.14,'B',transform=sub.transAxes,fontsize=panel_label_size,fontweight='bold')
        self._finish(fig,f'{output_name}_{path_mode}',show,save,dpi)
        return selected

