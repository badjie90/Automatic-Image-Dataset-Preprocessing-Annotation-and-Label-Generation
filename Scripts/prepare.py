"""Validate generated artifacts, produce training caches and manifests, audit split similarity."""
import sys,csv,json,collections,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
from PIL import Image
from src.data import letterbox

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'

def cache(r):
    sid=r['image_id'];meta=json.loads((DATA/'metadata'/f'{sid}.json').read_text())
    im=np.array(Image.open(DATA/'images'/f'{sid}.png').convert('RGB'));mask=np.array(Image.open(DATA/'masks'/f'{sid}.png'))
    assert im.shape[:2]==mask.shape,(sid,im.shape,mask.shape)
    assert set(np.unique(mask))<={0,1,2,255},sid
    assert (mask<3).any(),sid
    rgb,y,pad=letterbox(im,mask)
    Image.fromarray(rgb).save(DATA/'training_cache/images'/f'{sid}.png')
    Image.fromarray(y).save(DATA/'training_cache/masks'/f'{sid}.png')
    return {**r,'image':f'images/{sid}.png','mask':f'masks/{sid}.png','label_status':'automatic_pseudo_label','human_reviewed':False,
            'privacy_reviewed':False,'usable':True,'valid_fraction':meta['valid_fraction'],'privacy_fraction':meta['privacy_fraction'],
            **{f'pixels_{k}':v for k,v in meta['class_pixels'].items()}}

def main():
    cv2.setNumThreads(1)
    rows=list(csv.DictReader((DATA/'selected.csv').open()))
    missing=[r['image_id'] for r in rows if not (DATA/'metadata'/f"{r['image_id']}.json").exists()]
    assert not missing,f'{len(missing)} annotations missing, e.g. {missing[:3]}'
    for sub in ['images','masks']:(DATA/'training_cache'/sub).mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:result=list(ex.map(cache,rows))
    # Reserve a contiguous part of validation for temperature fitting. Never fit on the test set.
    val=[r for r in result if r['split']=='val'];boundary=round(int(val[round(.7*len(val))]['frame_index'])/900)*900
    for r in val:
        idx=int(r['frame_index'])
        if abs(idx-boundary)<=150:r['split']='calibration_guard';r['usable']=False
        elif idx>boundary:r['split']='calibration'
    # dHash similarity is a review signal, not proof of a shared scene.
    # Exact duplicate bytes were already removed by audit.py across the entire corpus.
    train=[r for r in result if r['split']=='train'];hold=[r for r in result if r['split'] in ['val','calibration','test']]
    pairs=[]
    tb=np.array([list(bytes.fromhex(r['dhash'])) for r in train],np.uint8)
    pop=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    for h in hold:
        dist=pop[np.bitwise_xor(tb,np.frombuffer(bytes.fromhex(h['dhash']),np.uint8))].sum(1)
        for k in np.flatnonzero(dist<=2):pairs.append({'train_id':train[k]['image_id'],'holdout_id':h['image_id'],'holdout_split':h['split'],'dhash_distance':int(dist[k])})
    # Conservatively remove training-side near duplicate candidates from this exploratory experiment.
    exclude={p['train_id'] for p in pairs}
    for r in result:
        if r['image_id'] in exclude:r['split']='similarity_excluded';r['usable']=False
    sets={s:{r['group_id'] for r in result if r['split']==s} for s in ['train','val','calibration','test']}
    for i,s in enumerate(sets):
        for t in list(sets)[i+1:]:assert not sets[s]&sets[t],f'provisional block leakage between {s} and {t}'
    def csvwrite(path,rs,fields=None):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fields or list(rs[0]));w.writeheader();w.writerows(rs)
    csvwrite(DATA/'manifest.csv',result)
    csvwrite(ROOT/'reports/cross_split_similarity.csv',pairs,['train_id','holdout_id','holdout_split','dhash_distance'])
    # Prioritise low-confidence frames and include uniformly distributed scenes in each subset.
    review=[]
    for split in ['train','val','calibration','test']:
        rs=[r for r in result if r['split']==split]
        chosen={r['image_id']:r for r in sorted(rs,key=lambda r:r['valid_fraction'])[:20]}
        for i in np.linspace(0,len(rs)-1,min(30,len(rs))).astype(int):chosen[rs[i]['image_id']]=rs[i]
        review.extend({'image_id':r['image_id'],'split':split,'image':r['image'],'mask':r['mask'],'reviewed':False,'reviewer':'','notes':''} for r in chosen.values())
    csvwrite(DATA/'review_queue.csv',review)
    counts={s:sum(r['split']==s for r in result) for s in sorted({r['split'] for r in result})}
    stats={'counts':counts,'selected_count':len(rows),'annotation_type':'automatic_pseudo_labels','human_reviewed_count':0,
           'excluded_training_similarity_candidates':len(exclude),'cross_split_similarity_pairs':len(pairs),'split_independence':'unverified route/recording identity; provisional filename-order holdout',
           'calibration_boundary_index':boundary,'calibration_guard_frames_each_side':150,
           'class_pixels':{s:{str(c):sum(int(r[f'pixels_{c}']) for r in result if r['split']==s) for c in [0,1,2,255]} for s in counts},
           'input_geometry':{'height':384,'width':640,'resize':'aspect-preserving bilinear; symmetric padding','mask_resize':'nearest','ignore_id':255}}
    (ROOT/'reports/dataset_summary.json').write_text(json.dumps(stats,indent=2))
    # Self-contained local review index uses processed imagery only.
    cards='\n'.join(f'<figure><a href="../data/overlays/{r["image_id"]}.jpg"><img loading="lazy" width="480" src="../data/overlays/{r["image_id"]}.jpg"></a><figcaption>{r["image_id"]} · {r["split"]} · automatic label</figcaption></figure>' for r in review)
    (ROOT/'reports/review.html').write_text('<!doctype html><meta charset="utf-8"><title>Lisbon label review</title><h1>Automatic label review</h1><p>Green path · red obstacle · blue background · purple ignore. These masks and privacy processing have not been human-reviewed.</p><div style="display:flex;flex-wrap:wrap">'+cards+'</div>')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
