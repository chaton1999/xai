#!/usr/bin/env python3
"""Normal-mode feature cosine and magnitude: extract once, plot repeatedly.

For geometry tag ckpt_X compare matched frames from:
 normal_modes/nm_ckpt_X/modes_ckpt_X/mode_oh/npz_1/
 normal_modes/nm_ckpt_X/modes_ckpt_X/mode_oh/npz_1000000/

Cosine is evaluated between the two checkpoint feature vectors at each mode coordinate.
"""
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

class NormalModesAnalysis:
    ATOMS = ('C1','C2','H3','C4','H5','H6','O7','H8','O9')
    OLD_INDICES = (5,4,7,3,6,8,0,2,1)
    COLORS = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple')
    def __init__(self, xai_root=None):
        self.root = Path(xai_root or Path.home()/'XAI').expanduser().resolve()
        self.source = self.root/'normal_modes'
        self.destination = self.root/'data'/'normal_modes'

    def _folder(self, geometry_tag, checkpoint, mode):
        stem = self.source/f'nm_{geometry_tag}'/f'modes_{geometry_tag}'/mode/f'npz_{checkpoint}'
        if stem.is_dir():
            return stem
        # Catch directory naming variations without silently selecting wrong mode.
        search = sorted((self.source/f'nm_{geometry_tag}').glob(f'*/{mode}/npz_{checkpoint}'))
        if len(search)==1:
            return search[0]
        raise FileNotFoundError(f'Normal-mode data directory missing or ambiguous: {stem}\nMatches: {search}')

    @staticmethod
    def _frame_map(folder):
        found={}
        for f in folder.glob('mp2_perblock_analysis_*.npz'):
            m=re.search(r'analysis_(\d+)\.npz$',f.name)
            if m:
                i=int(m.group(1))
                if i in found:
                    raise ValueError(f'Duplicate frame {i} in {folder}')
                found[i]=f
        if not found:
            raise FileNotFoundError(f'No mp2_perblock_analysis_*.npz in {folder}')
        return found

    @staticmethod
    def _features(path, key, n_modules):
        with np.load(path,allow_pickle=False) as dat:
            if key not in dat:
                raise KeyError(f'Missing {key!r} in {path}; keys: {dat.files}')
            arr=np.asarray(dat[key],dtype=float)
        if arr.ndim!=3 or arr.shape[1]!=9:
            raise ValueError(f'Expected modules x 9 atoms x features in {path}; got {arr.shape}')
        if arr.shape[0]==n_modules+1:
            arr=arr[1:] # optional embedding, not a physical interaction module
        if arr.shape[0]!=n_modules:
            raise ValueError(f'Expected {n_modules} modules (or embedding+modules), got {arr.shape[0]} in {path}')
        return arr

    def extract(self, geometry_tag='ckpt_1', mode='mode_oh', checkpoint_a=1,
                checkpoint_b=1000000, frames=1000, amplitude=0.5,
                path_type='linear', feature_key='act', n_modules=5,
                progress_every=100):
        a=self._frame_map(self._folder(geometry_tag,checkpoint_a,mode))
        b=self._frame_map(self._folder(geometry_tag,checkpoint_b,mode))
        matched=sorted(set(a)&set(b))
        if not matched:
            raise ValueError('No matching frame numbers between the two checkpoint directories')
        if max(matched)>frames or min(matched)<1:
            raise ValueError('Frame index is outside 1..frames; check FRAMES and input files')
        if len(matched)!=frames:
            print(f'Warning: {len(matched)} matched frames of expected {frames}. Missing frames are not interpolated.')
        folder=self.destination/geometry_tag/mode
        folder.mkdir(parents=True,exist_ok=True)
        out=folder/f'comparison_{checkpoint_a}_vs_{checkpoint_b}.dat'
        rows=[]
        t0=time.perf_counter()
        for count,i in enumerate(matched,1):
            va=self._features(a[i],feature_key,n_modules)
            vb=self._features(b[i],feature_key,n_modules)
            if va.shape!=vb.shape:
                raise ValueError(f'Frame {i} feature shapes differ: {va.shape} vs {vb.shape}')
            na=np.linalg.norm(va,axis=-1)
            nb=np.linalg.norm(vb,axis=-1)
            cosine=np.divide(np.sum(va*vb,axis=-1),na*nb,
                             out=np.full_like(na,np.nan),where=(na*nb)>0)
            if path_type=='linear':
                x=-amplitude+2*amplitude*(i-1)/(frames-1)
            elif path_type=='sin':
                x=amplitude*np.sin(2*np.pi*(i-1)/frames)
            else:
                raise ValueError("path_type must be 'linear' or 'sin'")
            for label,old in zip(self.ATOMS,self.OLD_INDICES):
                row={'frame':i,'mode_coordinate':x,'atom':label}
                for m in range(n_modules):
                    row[f'P_M{m+1}']=cosine[m,old]
                    row[f'mag_A_M{m+1}']=na[m,old]
                    row[f'mag_B_M{m+1}']=nb[m,old]
                rows.append(row)
            if progress_every and (count%progress_every==0 or count==len(matched)):
                elapsed=max(time.perf_counter()-t0,1e-9)
                rate=count/elapsed
                print(f'Extraction {count}/{len(matched)} ({100*count/len(matched):.1f}%) '
                      f'{rate:.1f} frames/s ETA {(len(matched)-count)/rate:.0f}s',flush=True)
        table=pd.DataFrame(rows)
        table.to_csv(out,sep=' ',index=False,float_format='%.12g')
        print('Saved:',out)
        return table

    def load(self,geometry_tag='ckpt_1',mode='mode_oh',checkpoint_a=1,checkpoint_b=1000000):
        file=self.destination/geometry_tag/mode/f'comparison_{checkpoint_a}_vs_{checkpoint_b}.dat'
        if not file.is_file():
            raise FileNotFoundError(f'{file}\nRun EXTRACT=True first.')
        return pd.read_csv(file,sep=r'\s+')

    @staticmethod
    def _smooth(y,window):
        if window<=1:return y
        # Centered rolling mean preserving array length (also even windows).
        return pd.Series(y).rolling(int(window),center=True,min_periods=1).mean().to_numpy()

    def plot(self,table=None,kind='both',geometry_tag='ckpt_1',mode='mode_oh',
             checkpoint_a=1,checkpoint_b=1000000,module_magnitude=5,
             show=True,save=True,output_name='normal_mode_features',
             figsize=None,xlim=(-0.5,0.5),ylim_cos=(0.80,1.01),ylim_mag=None,
             module_colors=None,magnitude_colors=('tab:blue','tab:red'),
             label_a='Ckpt 1',label_b='Ckpt 10$^6$',
             cosine_scatter_modules=(1,5),cosine_line_modules=(2,3,4),
             scatter_alpha=0.32,scatter_size=9,line_width=2.0,
             smoothing_window=10,show_legend=True,legend_ncol=5,
             legend_x=0.5,legend_y=0.99,legend_size=13,
             label_size=19,tick_size=12,title_size=12,
             spine_width=1.8,major_tick_length=6,minor_tick_length=3,
             left=0.11,right=0.98,bottom=0.09,top=0.85,
             wspace=0.12,hspace=0.32,dpi=300):
        if kind not in ('cosine','magnitude','both'):
            raise ValueError("kind must be 'cosine', 'magnitude', or 'both'")
        if table is None:
            table=self.load(geometry_tag,mode,checkpoint_a,checkpoint_b)
        if module_colors is None: module_colors=self.COLORS
        if figsize is None: figsize=(15,13) if kind=='both' else (15,7)
        kinds=['cosine','magnitude'] if kind=='both' else [kind]
        nrows=3*len(kinds)
        fig,axes=plt.subplots(nrows,3,figsize=figsize,squeeze=False)
        # Figure C: 3x3; Figure D: 3x3; shared legend on top.
        legend_handles=[]
        for block,which in enumerate(kinds):
            for pos,atom in enumerate(self.ATOMS):
                r,c=divmod(pos,3)
                ax=axes[block*3+r,c]
                sub=table.loc[table['atom']==atom].sort_values('mode_coordinate')
                x=sub['mode_coordinate'].to_numpy(dtype=float)
                if which=='cosine':
                    for m in range(1,6):
                        key=f'P_M{m}'
                        if key not in sub:raise KeyError(f'Missing {key} in staged data')
                        y=sub[key].to_numpy(dtype=float)
                        color=module_colors[m-1]
                        if m in cosine_scatter_modules:
                            ax.scatter(x,y,c=color,s=scatter_size,alpha=scatter_alpha,zorder=2)
                        if m in cosine_line_modules or m in cosine_scatter_modules:
                            ax.plot(x,self._smooth(y,smoothing_window),
                                    color=color,ls='-' if m in cosine_scatter_modules else '--',
                                    lw=line_width,zorder=3)
                    if ylim_cos is not None:ax.set_ylim(*ylim_cos)
                else:
                    for letter,color in [('A',magnitude_colors[0]),('B',magnitude_colors[1])]:
                        key=f'mag_{letter}_M{module_magnitude}'
                        if key not in sub:raise KeyError(f'Missing {key} in staged data')
                        y=sub[key].to_numpy(dtype=float)
                        ax.scatter(x,y,color=color,alpha=scatter_alpha,s=scatter_size,zorder=2)
                        ax.plot(x,self._smooth(y,smoothing_window),color=color,lw=line_width,zorder=3)
                    local_lim=ylim_mag.get(atom) if isinstance(ylim_mag,dict) else ylim_mag
                    if local_lim is not None:ax.set_ylim(*local_lim)
                if xlim is not None:ax.set_xlim(*xlim)
                ax.set_title(f'Atom {atom}',fontsize=title_size,pad=2)
                if c==0:
                    ax.set_ylabel(r'$\mathcal{P}^{\alpha}$' if which=='cosine'
                                  else r'$\|\mathcal{F}^{\alpha}\|$',fontsize=label_size)
                else:
                    ax.tick_params(labelleft=False)
                if r==2:
                    ax.set_xlabel(r'Mode coordinate $a$ ($\AA$)',fontsize=label_size)
                    ax.tick_params(axis='x',labelrotation=45)
                else:
                    ax.tick_params(labelbottom=False)
                ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                ax.tick_params(which='major',labelsize=tick_size,width=spine_width,length=major_tick_length,direction='in')
                ax.tick_params(which='minor',width=spine_width*.7,length=minor_tick_length,direction='in')
                for spine in ax.spines.values():spine.set_linewidth(spine_width)
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace,hspace=hspace)
        if show_legend:
            if 'cosine' in kinds:
                handles=[Line2D([0],[0],color=module_colors[m],lw=line_width,
                                ls='-' if m+1 in cosine_scatter_modules else '--',label=f'M{m+1}')
                         for m in range(5)]
                fig.legend(handles=handles,ncol=legend_ncol,loc='upper center',
                           bbox_to_anchor=(legend_x,legend_y),frameon=False,fontsize=legend_size)
            elif 'magnitude' in kinds:
                handles=[Line2D([0],[0],color=magnitude_colors[i],lw=line_width,label=l)
                         for i,l in enumerate((label_a,label_b))]
                fig.legend(handles=handles,ncol=2,loc='upper center',
                           bbox_to_anchor=(legend_x,legend_y),frameon=False,fontsize=legend_size)
        if save:
            dest=self.destination/geometry_tag/mode
            dest.mkdir(parents=True,exist_ok=True)
            prefix=f'{output_name}_{kind}'
            for suffix in ('png','pdf'):
                path=dest/f'{prefix}.{suffix}'
                fig.savefig(path,dpi=dpi,bbox_inches='tight')
                print('Saved:',path)
        if show:plt.show()
        plt.close(fig)

