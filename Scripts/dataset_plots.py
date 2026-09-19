"""Measured selection coverage and class composition, without claims of citywide coverage."""
import csv,json,sys,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'reports/.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
def main():
    data=ROOT/'data';out=ROOT/'reports';raw=list(csv.DictReader((data/'inventory.csv').open()));sel=list(csv.DictReader((data/'manifest.csv').open()))
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,key,title in zip(axs,['brightness','laplacian_variance','darkness_fraction'],['Mean luminance','Laplacian variance at 160 × 90','Dark-pixel fraction']):
        a=np.array([float(r[key]) for r in raw if r['status']=='valid']);b=np.array([float(r[key]) for r in sel]);bins=np.linspace(min(a.min(),b.min()),max(a.max(),b.max())+1e-6,40)
        ax.hist(a,bins=bins,density=True,histtype='step',label='All raw images');ax.hist(b,bins=bins,density=True,histtype='step',label='Selected images');ax.set(xlabel=title,ylabel='Density');ax.legend(fontsize=8)
    fig.suptitle('Measured coverage of temporal sampling');fig.tight_layout();fig.savefig(out/'sampling_coverage.png');plt.close(fig)
    stats=json.loads((out/'dataset_summary.json').read_text());splits=['train','val','calibration','test'];colors=['#00a640','#cf3333','#465a82','#aa00aa']
    fig,axs=plt.subplots(1,2,figsize=(11,4));bottom=np.zeros(4)
    for c,name in enumerate(['path','obstacle','background','ignore']):
        cid=255 if c==3 else c;vals=[]
        for s in splits:
            counts=stats['class_pixels'][s];vals.append(counts[str(cid)]/sum(counts.values()))
        axs[0].bar(splits,vals,bottom=bottom,label=name,color=colors[c]);bottom+=vals
    axs[0].set(ylabel='Fraction of native-resolution pixels',title='Automatic class composition');axs[0].legend(fontsize=8)
    for i,s in enumerate(splits):
        rs=[r for r in sel if r['split']==s];axs[1].scatter([int(r['frame_index']) for r in rs],[i]*len(rs),s=2)
    axs[1].set(yticks=range(4),yticklabels=splits,xlabel='Source filename frame index',title='Provisional chronological partition')
    fig.tight_layout();fig.savefig(out/'dataset_composition.png');plt.close(fig)
    rows=[]
    for r in sel:
        rows.append({'image_id':r['image_id'],'split':r['split'],'label_type':'automatic_pseudo_label',**{f'contains_{n}':int(r[f'pixels_{c}'])>0 for c,n in enumerate(['path','obstacle','background'])},'ignored_pixels':r['pixels_255']})
    with (data/'class_presence.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    canvas=Image.new('RGB',(1200,980),'white');draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(np.linspace(0,len(sel)-1,12).astype(int)):
        r=sel[i];im=Image.open(data/'overlays'/f"{r['image_id']}.jpg");im.thumbnail((400,225))
        x=(j%3)*400;y=(j//3)*245;canvas.paste(im,(x,y));draw.text((x+5,y+228),f"{r['image_id']}  {r['split']}",fill='black')
    canvas.save(out/'annotation_examples.jpg',quality=92)

if __name__=='__main__':main()
