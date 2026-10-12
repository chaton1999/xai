#!/usr/bin/env python3
"""Extract and plot molecule-wise interaction KRM similarity across TL checkpoints.

Sources: ~/XAI/project_feature/spearman/atom_wise/md_{yes,no}/
         krm_interaction_molecule/ckpt_<N>/module_<M>.npz
Output:  ~/XAI/data/spearman/metrics_ref_<N>.dat

FULL_FEATURE_PICTURE=False: only module 5; True: all modules 1..5.
The underlying KRM y is an upper-triangle vector (500*499/2=124750).
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, LogFormatterMathtext, NullFormatter

CHECKPOINTS = (1,5,10,15,20,25,30,50,75,100,200,300,400,500,1000,5000,10000,100000,200000,500000,750000,1000000)

class XAISpearman:
    def __init__(self, xai_root=None):
        self.root = Path(xai_root or Path.home()/'XAI').expanduser().resolve()
        self.source = self.root/'project_feature/spearman/atom_wise'
        self.outdir = self.root/'data/spearman'

    def _source_file(self, mode, ckpt, module):
        d = {'true':'md_yes', 'false':'md_no'}[mode]
        return self.source/d/'krm_interaction_molecule'/f'ckpt_{ckpt}'/f'module_{module}.npz'

    @staticmethod
    def _read_vector(path):
        if not path.is_file():
            raise FileNotFoundError(f'KRM missing: {path}')
        with np.load(path, allow_pickle=False) as npz:
            if 'y' in npz.files:
                y = np.asarray(npz['y'], dtype=np.float64).ravel()
            elif 'G' in npz.files:
                g = np.asarray(npz['G'], dtype=np.float64)
                if g.ndim != 2 or g.shape[0] != g.shape[1]:
                    raise ValueError(f'G is not square: {path}')
                y = g[np.triu_indices(g.shape[0], k=1)]
            else:
                raise KeyError(f'Neither y nor G in {path}')
            meta = {}
            for name in ('frame_ids','structure_stride','n_structures','representation_level','feature_key','use_cosine_within_G','feature_dim'):
                if name in npz.files:
                    meta[name] = np.asarray(npz[name]).copy()
        if not np.isfinite(y).all():
            raise ValueError(f'Nonfinite KRM y values: {path}')
        return y, meta

    @staticmethod
    def _rank(v):
        # scipy is already used in the user's source scripts; average ties.
        from scipy.stats import rankdata
        return rankdata(v, method='average')

    @staticmethod
    def _cos(a,b):
        den = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/den) if den > 0 else float('nan')

    def extract(self, reference_checkpoint=1000000, full_feature_picture=False,
                checkpoints=CHECKPOINTS, progress_every=5):
        modules = range(1,6) if full_feature_picture else (5,)
        checkpoints = tuple(int(x) for x in checkpoints)
        needed = [self._source_file(mode,ck,mod) for mode in ('true','false') for mod in modules for ck in checkpoints]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} expected files missing; examples:\n'+'\n'.join(missing[:12]))
        self.outdir.mkdir(parents=True, exist_ok=True)
        rows=[]
        count=0; total=len(needed); t0=time.perf_counter()
        for mode in ('true','false'):
            for module in modules:
                ref_path = self._source_file(mode, reference_checkpoint, module)
                ref, refmeta = self._read_vector(ref_path)
                rref = self._rank(ref)
                for ckpt in checkpoints:
                    path = self._source_file(mode, ckpt, module)
                    if ckpt == reference_checkpoint:
                        vec, metadata = ref, refmeta
                    else:
                        vec, metadata = self._read_vector(path)
                    if vec.shape != ref.shape:
                        raise ValueError(f'KRM vector length mismatch at {path}: {vec.shape} vs {ref.shape}')
                    for key in ('frame_ids','structure_stride','n_structures','representation_level','feature_key','use_cosine_within_G','feature_dim'):
                        if key in refmeta and key in metadata and not np.array_equal(refmeta[key],metadata[key]):
                            raise ValueError(f'{key} mismatch for {path}; cannot compare unmatched KRMs')
                    if ckpt == reference_checkpoint:
                        c=s=1.0
                    else:
                        c=self._cos(vec,ref)
                        # Spearman = Pearson of the two centered rank vectors.
                        ra=self._rank(vec); rb=rref
                        s=self._cos(ra-ra.mean(),rb-rb.mean())
                    rows.append(dict(checkpoint=ckpt,model=mode,module=module,cosine=c,spearman=s))
                    count += 1
                    if progress_every and (count==1 or count%progress_every==0 or count==total):
                        elapsed=max(time.perf_counter()-t0,1e-9)
                        rate=count/elapsed
                        print(f'\rExtraction {count}/{total} ({count/total:.1%}) | {rate:.2f} files/s | ETA {(total-count)/rate:.0f}s',end='\n' if count==total else '',flush=True)
        df=pd.DataFrame(rows).sort_values(['model','module','checkpoint']).reset_index(drop=True)
        file=self.outdir/f'metrics_ref_{reference_checkpoint}.dat'
        if file.exists():
            previous=pd.read_csv(file,sep=r'\s+')
            previous=previous[~previous.set_index(['model','module','checkpoint']).index.isin(df.set_index(['model','module','checkpoint']).index)]
            df=pd.concat([previous,df],ignore_index=True).sort_values(['model','module','checkpoint']).reset_index(drop=True)
        df.to_csv(file,sep=' ',index=False,float_format='%.12g')
        print(f'\nSaved {len(df)} rows: {file}')
        return df

    def load(self,reference_checkpoint=1000000):
        file=self.outdir/f'metrics_ref_{reference_checkpoint}.dat'
        if not file.is_file():
            raise FileNotFoundError(f'{file} missing: extract first')
        return pd.read_csv(file,sep=r'\s+')

    def plot(self, table=None, reference_checkpoint=1000000,
             full_feature_picture=False, mode='both', show_plot=True, save_plot=True,
             figsize=None, xlim=(0.8,1.2e6), ylimits=None,
             colors=('blue','red','green','purple','orange'), markers=('o','s','^','D','v'),
             linewidth=2.8, markersize=6, label_size=22,tick_size=18,
             title_size=21,legend_size=17,spine_width=2.5,
             major_tick_length=9,minor_tick_length=5, major_tick_width=2.5,minor_tick_width=1.5,
             left=0.13,right=0.96,bottom=0.11,top=0.84,wspace=0.16,hspace=0.22,
             legend_ncol=5, legend_x=0.52, legend_y=0.985,
             show_legend=True, show_grid=False, dpi=300,
             output_name='interaction_krm_spearman_cosine'):
        if mode not in ('both','true','false'):
            raise ValueError('mode must be both, true, or false')
        df=self.load(reference_checkpoint) if table is None else table.copy()
        modules=tuple(range(1,6)) if full_feature_picture else (5,)
        modes=('true','false') if mode=='both' else (mode,)
        metrics=('spearman','cosine')
        subset=df[df['model'].isin(modes)&df['module'].isin(modules)]
        for m in modes:
            for k in modules:
                if subset[(subset.model==m)&(subset.module==k)].empty:
                    raise ValueError(f'Missing staged data: mode={m}, module={k}. Run extract(full_feature_picture=True) to stage M1-M5.')
        ncols=len(modes)
        if figsize is None: figsize=(14,11) if ncols==2 else (8,11)
        fig,axes=plt.subplots(2,ncols,figsize=figsize,squeeze=False,sharex=True,sharey='row' if ncols==2 else False)
        handles={}
        for row,metric in enumerate(metrics):
            for col,m in enumerate(modes):
                ax=axes[row,col]
                for k in modules:
                    d=subset[(subset.model==m)&(subset.module==k)].sort_values('checkpoint')
                    line,=ax.plot(d['checkpoint'],d[metric],color=colors[k-1],marker=markers[k-1],linewidth=linewidth,markersize=markersize,label=f'M{k}')
                    handles[k]=line
                ax.set_xscale('log'); ax.set_xlim(*xlim)
                if ylimits is not None and metric in ylimits and ylimits[metric] is not None:
                    ax.set_ylim(*ylimits[metric])
                ax.xaxis.set_major_locator(LogLocator(base=10,numticks=8))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(which='major',labelsize=tick_size,length=major_tick_length,width=major_tick_width,direction='inout')
                ax.tick_params(which='minor',length=minor_tick_length,width=minor_tick_width,direction='inout')
                for spine in ax.spines.values(): spine.set_linewidth(spine_width)
                ax.grid(True,alpha=0.2) if show_grid else ax.grid(False)
                if col==0:
                    ax.set_ylabel(r'$\rho(y^{N},y^{10^{6}})$' if metric=='spearman' else r'$S(y^{N},y^{10^{6}})$',fontsize=label_size,math_fontfamily='cm')
                if row==0:
                    ax.set_title(r'$E_{\mathrm{elec}}=\mathrm{True}$' if m=='true' else r'$E_{\mathrm{elec}}=\mathrm{False}$',fontsize=title_size)
                if row==1:
                    ax.set_xlabel(r'Transfer Learning Steps $N$',fontsize=label_size)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
        if show_legend:
            fig.legend([handles[k] for k in modules],[f'M{k}' for k in modules],loc='upper center',bbox_to_anchor=(legend_x,legend_y),ncol=legend_ncol,frameon=False,fontsize=legend_size)
        if save_plot:
            self.outdir.mkdir(parents=True,exist_ok=True)
            suffix='all_modules' if full_feature_picture else 'm5'
            base=self.outdir/f'{output_name}_{suffix}_{mode}_ref_{reference_checkpoint}'
            for ext in ('png','pdf'):
                fig.savefig(f'{base}.{ext}',dpi=dpi,bbox_inches='tight')
            print(f'Saved: {base}.png and .pdf')
        if show_plot: plt.show()
        plt.close(fig)

    # ============================================================
    # SUPPORTING DATA: MAE AND ELECTROSTATIC ENERGIES
    # ============================================================

    def extract_auxiliary(self, checkpoints=CHECKPOINTS, frame_min=1,
                          frame_max=1000, frame_stride=1, num_blocks=5,
                          series_root=None, strict_elec=False,
                          elec_source="auto", progress_every=5):
        """Stage EMAE (kcal/mol) and mean electrostatic energy (eV).

        elec_source:
          auto   -> prefer existing md_yes/<ckpt>/mp2_perblock_analysis_<frame>.npz;
                    fall back to saved frame_series_ckpt_<ckpt>.dat
          npz    -> only use the existing per-frame NPZ files
          series -> only use the frame_series files

        Never runs PhysNet; never silently substitutes zero for missing values.
        Select frames by their original 1-based frame IDs, inclusively.
        """
        if elec_source not in ("auto", "npz", "series"):
            raise ValueError("elec_source must be 'auto', 'npz', or 'series'")
        if frame_stride < 1 or frame_max < frame_min:
            raise ValueError("Invalid frame selection")
        self.outdir.mkdir(parents=True, exist_ok=True)
        checkpoints = tuple(int(ck) for ck in checkpoints)
        combined = self.root / 'data/combined_9columns.dat'
        if not combined.is_file():
            raise FileNotFoundError(f'MAE table missing: {combined}')
        mae = pd.read_csv(combined, sep=r'\s+', comment='#')
        required = {'checkpoint', 'EMAE_true', 'EMAE_false'}
        if not required.issubset(mae.columns):
            raise ValueError(f'MAE table missing columns: {required - set(mae.columns)}')
        mae = mae.set_index('checkpoint')

        import re
        series_bases = (
            [Path(series_root).expanduser()] if series_root is not None
            else [self.source / 'md_yes', self.source / 'md',
                  self.source / 'from_komeda' / 'md', self.source]
        )

        def mean_from_npz(ck):
            folder = self.source / 'md_yes' / str(ck)
            if not folder.is_dir():
                return np.nan, 0
            paths = {}
            for path in folder.glob('mp2_perblock_analysis_*.npz'):
                match = re.fullmatch(r'mp2_perblock_analysis_(\d+)\.npz', path.name)
                if match:
                    frame = int(match.group(1))
                    if frame_min <= frame <= frame_max and (frame-frame_min) % frame_stride == 0:
                        paths[frame] = path
            if not paths:
                return np.nan, 0
            values = []
            for frame, path in sorted(paths.items()):
                with np.load(path, allow_pickle=False) as data:
                    if 'E_elec_total' in data.files:
                        arr = np.asarray(data['E_elec_total'], dtype=float)
                        if arr.size != 1:
                            raise ValueError(f'Expected scalar E_elec_total in {path}, got {arr.shape}')
                        value = float(arr.item())
                    elif 'E_elec_atom' in data.files:
                        value = float(np.asarray(data['E_elec_atom'], dtype=float).sum())
                    else:
                        raise KeyError(f'No E_elec_total or E_elec_atom in {path}')
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite electrostatic energy in {path}')
                values.append(value)
            return float(np.mean(values)), len(values)

        def mean_from_series(ck):
            for base in series_bases:
                candidates = (
                    base / str(ck) / 'series_out' / f'frame_series_ckpt_{ck}.dat',
                    base / 'md' / str(ck) / 'series_out' / f'frame_series_ckpt_{ck}.dat',
                    base / 'series_out' / f'frame_series_ckpt_{ck}.dat',
                )
                for path in candidates:
                    if not path.is_file():
                        continue
                    arr = np.loadtxt(path, comments='#', ndmin=2)
                    energy_column = 1 + 2 * num_blocks
                    if arr.shape[1] <= energy_column:
                        raise ValueError(f'No column {energy_column} in {path}')
                    frame = arr[:, 0].astype(int)
                    selected = ((frame >= frame_min) & (frame <= frame_max)
                                & ((frame-frame_min) % frame_stride == 0))
                    if selected.any():
                        values = arr[selected, energy_column]
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite energies in {path}')
                        return float(np.mean(values)), int(selected.sum())
            return np.nan, 0

        rows = []
        for idx, ck in enumerate(checkpoints, start=1):
            if ck not in mae.index:
                raise ValueError(f'Checkpoint {ck} absent from {combined}')
            result, n_used, source = np.nan, 0, 'missing'
            if elec_source in ('auto', 'npz'):
                result, n_used = mean_from_npz(ck)
                if n_used:
                    source = 'npz'
            if not n_used and elec_source in ('auto', 'series'):
                result, n_used = mean_from_series(ck)
                if n_used:
                    source = 'series'
            rows.append({
                'checkpoint': ck,
                'EMAE_true_kcalmol': float(mae.loc[ck, 'EMAE_true']),
                'EMAE_false_kcalmol': float(mae.loc[ck, 'EMAE_false']),
                'mean_Eelec_eV': result,
                'n_elec_frames': n_used,
                'elec_source': source,
            })
            if progress_every and (idx == 1 or idx % progress_every == 0 or idx == len(checkpoints)):
                print(f'Electrostatic extraction: {idx}/{len(checkpoints)} checkpoints; latest {ck}: {n_used} frames ({source})', flush=True)

        df = pd.DataFrame(rows)
        outfile = self.outdir / 'auxiliary_mae_elec.dat'
        df.to_csv(outfile, sep=' ', index=False, na_rep='nan', float_format='%.12g')
        print(f'Saved: {outfile}')
        missing = df.loc[df['mean_Eelec_eV'].isna(), 'checkpoint'].tolist()
        if missing:
            print(f'Missing electrostatic data for checkpoints: {missing}')
            if strict_elec:
                raise FileNotFoundError('No selected E_elec_total values for those checkpoints')
        return df

    def load_auxiliary(self):
        path = self.outdir/'auxiliary_mae_elec.dat'
        if not path.is_file():
            raise FileNotFoundError(f'{path} missing: run extract_auxiliary()')
        return pd.read_csv(path, sep=r'\s+')

    def plot_metrics_with_energy(self, table=None, auxiliary=None,
                                 reference_checkpoint=1000000,
                                 mode='both', show_plot=True, save_plot=True,
                                 figsize=(14,11), xlim=(0.8,1.2e6),
                                 spearman_ylim=(0.98,1.0002),
                                 cosine_ylim=None, mae_ylim=None,
                                 elec_ylim=None, show_elec=True, show_mae=True,
                                 missing_elec='raise',
                                 colors=('darkred','red'), linewidth=2.8,
                                 markersize=7, label_size=21, tick_size=16,
                                 legend_size=13, spine_width=2.5,
                                 left=0.15,right=0.84,bottom=0.12,top=0.90,
                                 hspace=0.19, dpi=300, show_grid=False,
                                 show_legend=True, legend_ncol=2,
                                 legend_x=0.50, legend_y=1.01,
                                 output_name='m5_spearman_elec_cosine_mae'):
        """Two rows: Spearman+mean E_elec and cosine+EMAE.

        With missing_elec='raise', fails when electrostatics source is missing.
        With missing_elec='omit', plots Spearman without an electrostatic trace.
        Both correlation modes may be shown by mode='both'.
        """
        if mode not in ('both','true','false'):
            raise ValueError('mode must be both, true, or false')
        if missing_elec not in ('raise','omit'):
            raise ValueError('missing_elec must be raise or omit')
        metrics = self.load(reference_checkpoint) if table is None else table.copy()
        aux = self.load_auxiliary() if auxiliary is None else auxiliary.copy()
        aux = aux.set_index('checkpoint').sort_index()
        fig, axes = plt.subplots(2,1, figsize=figsize, sharex=True)
        modes = ('true','false') if mode == 'both' else (mode,)
        handles = []; labels=[]
        style = {'true':'-', 'false':'--'}
        for j,(ax,metric) in enumerate(zip(axes,('spearman','cosine'))):
            axr = ax.twinx() if ((j==0 and show_elec) or (j==1 and show_mae)) else None
            for model in modes:
                d = metrics[(metrics.model == model) & (metrics.module == 5)].sort_values('checkpoint')
                if d.empty:
                    raise ValueError(f'No M5 {model} metrics staged')
                line, = ax.plot(d.checkpoint, d[metric], color=colors[j],
                                linestyle=style[model], marker='x',
                                linewidth=linewidth, markersize=markersize,
                                label=f'M5 {metric}, E_elec={model}')
                handles.append(line); labels.append(line.get_label())
            if j == 0:
                if spearman_ylim is not None: ax.set_ylim(*spearman_ylim)
                ax.set_ylabel(r'$\rho(y^N,y^{10^6})$',fontsize=label_size, color=colors[j], math_fontfamily='cm')
                if show_elec:
                    missing = aux['mean_Eelec_eV'].isna()
                    if missing.any() and missing_elec=='raise':
                        plt.close(fig)
                        raise ValueError('Mean electrostatic energy not staged for all checkpoints. Check series files, or set MISSING_ELEC="omit".')
                    valid=aux.loc[~missing]
                    if len(valid):
                        line,=axr.plot(valid.index, valid.mean_Eelec_eV, color='black',
                                       linestyle='-',marker='s',linewidth=linewidth,
                                       markersize=markersize, label=r'$E_{\mathrm{elec}}$')
                        handles.append(line);labels.append(line.get_label())
                        if elec_ylim is not None: axr.set_ylim(*elec_ylim)
                    else:
                        axr.set_yticks([])
                    axr.set_ylabel(r'$E_{\mathrm{elec}}$ (eV)',fontsize=label_size)
            else:
                if cosine_ylim is not None: ax.set_ylim(*cosine_ylim)
                ax.set_ylabel(r'$S(y^N,y^{10^6})$',fontsize=label_size, color=colors[j], math_fontfamily='cm')
                if show_mae:
                    for model in modes:
                        col=f'EMAE_{model}_kcalmol'
                        line,=axr.plot(aux.index,aux[col],color='black',
                                       linestyle=style[model],marker='o',
                                       markerfacecolor=('black' if model=='true' else 'white'),
                                       markersize=markersize,linewidth=linewidth,
                                       label=f'EMAE, E_elec={model}')
                        handles.append(line); labels.append(line.get_label())
                    axr.set_yscale('log')
                    if mae_ylim is not None:axr.set_ylim(*mae_ylim)
                    axr.set_ylabel(r'MAE$(E)$ (kcal/mol)',fontsize=label_size)
            ax.set_xscale('log'); ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(LogLocator(base=10,numticks=8))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(True,alpha=.2) if show_grid else ax.grid(False)
            for aa in [ax]+([axr] if axr is not None else []):
                for spine in aa.spines.values(): spine.set_linewidth(spine_width)
                aa.tick_params(which='major',direction='in',labelsize=tick_size,length=9,width=2)
                aa.tick_params(which='minor',direction='in',length=5,width=1.5)
            ax.tick_params(axis='y',colors=colors[j])
        axes[-1].set_xlabel(r'Transfer Learning Steps $N$',fontsize=label_size)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,hspace=hspace)
        if show_legend:
            fig.legend(handles,labels,ncol=legend_ncol,loc='upper center',
                       bbox_to_anchor=(legend_x,legend_y),frameon=False,fontsize=legend_size)
        if save_plot:
            self.outdir.mkdir(parents=True,exist_ok=True)
            base=self.outdir/f'{output_name}_{mode}_ref_{reference_checkpoint}'
            for ext in ('png','pdf'):
                fig.savefig(f'{base}.{ext}',dpi=dpi,bbox_inches='tight')
            print(f'Saved {base}.png and .pdf')
        if show_plot: plt.show()
        plt.close(fig)

