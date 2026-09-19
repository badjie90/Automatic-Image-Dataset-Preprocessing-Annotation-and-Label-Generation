"""Read-only full inventory, integrity checks, and deterministic temporal sampling."""
import argparse, collections, concurrent.futures, csv, hashlib, io, json, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def inspect(path):
    r = {'image_id': path.stem, 'source': str(path.resolve()), 'frame_index': int(path.stem.split('_')[-1])}
    try:
        raw = path.read_bytes()
        r.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            r.update(width=im.width, height=im.height, mode=im.mode)
            if im.mode == 'RGBA':
                alpha=im.getchannel('A'); r['alpha_min'],r['alpha_max']=alpha.getextrema()
            rgb=np.array(im.convert('RGB').resize((160,90)))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        r.update(brightness=float(gray.mean()), darkness_fraction=float((gray<10).mean()),
                 saturation_fraction=float((gray>245).mean()), laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                 dhash=''.join(f'{int(v):02x}' for v in np.packbits(cv2.resize(gray,(9,8))[:,1:]>cv2.resize(gray,(9,8))[:,:-1])),
                 status='valid', error='')
    except Exception as e:
        r.update(status='unreadable',error=str(e))
    return r

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--stride',type=int,default=15);a.add_argument('--workers',type=int,default=12);args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    paths=sorted(args.source.glob('*.png')); rows=[]; start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i,r in enumerate(ex.map(inspect,paths)):
            rows.append(r)
            if (i+1)%2000==0: print(f'audited {i+1}/{len(paths)} in {time.time()-start:.1f}s',flush=True)
    # Keep every raw file. Exclude only unreadable images and exact byte duplicates from the candidate set.
    hashes={}; valid=[]
    for r in rows:
        if r['status']=='valid':
            if r['sha256'] in hashes:r['status']='exact_duplicate';r['duplicate_of']=hashes[r['sha256']]
            else:hashes[r['sha256']]=r['image_id'];valid.append(r)
    maxidx=max(r['frame_index'] for r in rows); cuts=[int(.70*maxidx),int(.85*maxidx)];guard=300
    selected=[]
    for r in rows:
        r['selected']=False;r['split']='unused';r['group_id']=f"proxy_block_{(r['frame_index']-1)//900:04d}"
        if r['status']!='valid': continue
        if any(abs(r['frame_index']-cut)<=guard for cut in cuts):r['split']='temporal_guard';continue
        if (r['frame_index']-1)%args.stride:continue
        r['selected']=True
        r['split']='train' if r['frame_index']<cuts[0] else ('val' if r['frame_index']<cuts[1] else 'test')
        selected.append(r)
    for name,items in [('inventory.csv',rows),('selected.csv',selected)]:
        fields=sorted(set().union(*(r.keys() for r in items)))
        with (args.out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(items)
    summary={'raw_count':len(rows),'raw_bytes':sum(r.get('bytes',0) for r in rows),'status_counts':dict(collections.Counter(r['status'] for r in rows)),
             'dimensions':dict(collections.Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)),
             'selected_count':len(selected),'split_counts':dict(collections.Counter(r['split'] for r in selected)),
             'stride':args.stride,'nominal_sampling_fps':30/args.stride,'split_boundaries_frame_index':cuts,'guard_frames_each_side':guard,
             'split_status':'provisional contiguous filename-order holdout; recording and route independence unverified',
             'quality_policy':'report brightness, exposure and sharpness; no arbitrary rejection of challenging scenes',
             'elapsed_seconds':time.time()-start}
    (args.out/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
