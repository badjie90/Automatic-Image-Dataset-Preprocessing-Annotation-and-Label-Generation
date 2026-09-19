"""Mapillary teacher proposals, conservative region blurring, and reproducible exports.

These are automatic labels. No output from this program is human-reviewed ground truth.
"""
import argparse,csv,json,sys,time,hashlib,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation

PATH_IDS=[7,8,9,10,11,13,14,15,23,24,36,41]
BACKGROUND_IDS=[16,18,25,26,27,28,29,30,31,32]
IGNORE_IDS=[63,64]
PRIVACY_IDS=[19,20,21,22,54,55,56,57,58,59,60,61,62]
PALETTE=np.array([[0,190,60],[230,45,45],[70,90,130]],dtype=np.uint8)

def save_png(path,array):
    Image.fromarray(array).save(path)

def policy_fingerprint(policy):
    plain=json.loads(json.dumps(policy));plain.pop('policy_sha256',None)
    return hashlib.sha256(json.dumps(plain,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--limit',type=int,default=0);ap.add_argument('--manifest',default='selected.csv')
    ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--device',default='cuda:0');ap.add_argument('--long-edge',type=int,default=1024)
    ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();root=args.root.resolve();out=root/'data';torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=list(csv.DictReader((out/args.manifest).open()));rows=rows[:args.limit] if args.limit else rows
    rows=rows[args.shard_index::args.num_shards]
    for name in ['images','masks','dense_proposals','teacher_labels','confidence','privacy_masks','overlays','metadata']:(out/name).mkdir(exist_ok=True)
    checkpoint=root/'checkpoints/teacher'
    mapping=np.ones(65,dtype=np.uint8);mapping[PATH_IDS]=0;mapping[BACKGROUND_IDS]=2;mapping[IGNORE_IDS]=255
    policy={'classes':{'0':'path','1':'obstacle','2':'background','255':'ignore'},'teacher_id2label':json.loads((checkpoint/'config.json').read_text())['id2label'],
            'teacher_to_target':mapping.tolist(),'privacy_teacher_ids':PRIVACY_IDS,'privacy_policy':'blur full predicted people/riders and motor vehicles conservatively, dilate 9 pixels, Gaussian kernel 71 at original resolution',
            'label_status':'automatic_pseudo_label','threshold':.70,'confidence_note':'normalised aggregated teacher semantic scores, not calibrated probabilities',
            'path_policy':'roads, bike lanes, crossings, sidewalks, pedestrian areas, parking, service lanes, curb cuts and ground fixtures',
            'background_policy':'bridges, tunnels, natural terrain, sky, vegetation, water and banners; not a traversable class',
            'ignore_policy':'ego vehicle/car mount, low teacher confidence, transparent source pixels',
            'teacher_long_edge':args.long_edge,'teacher_config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
            'teacher_preprocessor_config_sha256':hashlib.sha256((checkpoint/'preprocessor_config.json').read_bytes()).hexdigest(),
            'teacher_weights_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            'pipeline_version':'1.0-probability-interpolation',
            'postprocessing':'sigmoid query masks before bilinear interpolation; combine semantic scores, restore native size, aggregate target classes, normalise score mass',
            'resume_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    policy['policy_sha256']=policy_fingerprint(policy)
    policy_path=out/'label_policy.json'
    if policy_path.exists():
        old=json.loads(policy_path.read_text())
        if old.get('policy_sha256')!=policy['policy_sha256'] or policy_fingerprint(old)!=policy['policy_sha256']:
            raise RuntimeError('Annotation policy, source implementation, or weights differ. Use a new versioned dataset directory; existing masks will not be relabelled or misdescribed.')
    elif args.shard_index==0:
        tmp=out/'label_policy.tmp';tmp.write_text(json.dumps(policy,indent=2));tmp.replace(policy_path)
    start=time.time();done=0
    pending=[]
    for r in rows:
        sid=r['image_id'];meta_path=out/'metadata'/f'{sid}.json'
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta.get('annotation_policy_sha256')!=policy['policy_sha256'] or meta.get('sha256')!=r.get('sha256'):
                raise RuntimeError(f'Provenance mismatch for {sid}; refusing to mix annotation versions.')
        files=[out/sub/f'{sid}.png' for sub in ['images','masks','dense_proposals','teacher_labels','confidence','privacy_masks']]+[out/'overlays'/f'{sid}.jpg',meta_path]
        if not all(p.is_file() and p.stat().st_size>0 for p in files):pending.append(r)
    print(f'Policy verified; complete {len(rows)-len(pending)}, pending {len(pending)}',flush=True)
    if args.verify_only or not pending:return
    processor=AutoImageProcessor.from_pretrained(checkpoint,local_files_only=True)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint,local_files_only=True).to(args.device).eval()
    for offset in range(0,len(pending),args.batch_size):
        batch=pending[offset:offset+args.batch_size]; originals=[];small=[];alphas=[]
        for r in batch:
            raw=Path(r['source']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:
                raise RuntimeError(f"Source content differs from manifest for {r['image_id']}; refusing to annotate changed input.")
            with Image.open(io.BytesIO(raw)) as im:
                alphas.append(np.array(im.getchannel('A')) if im.mode=='RGBA' else None)
                rgb=im.convert('RGB');originals.append(np.array(rgb));rgb.thumbnail((args.long_edge,args.long_edge));small.append(rgb)
        # Input sizes are multiples of 32 after padding. Preserve each image's aspect ratio.
        inputs=processor(images=small,return_tensors='pt',do_resize=False).to(args.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):res=model(**inputs)
        class_probs=res.class_queries_logits.float().softmax(-1)[...,:-1]
        masks=res.masks_queries_logits.float().sigmoid()
        # Aggregate at the padded input resolution, then crop padding before restoring original coordinates.
        full=F.interpolate(masks,size=inputs.pixel_values.shape[-2:],mode='bilinear',align_corners=False)
        for b,(r,im,thumb,alpha) in enumerate(zip(batch,originals,small,alphas)):
            h,w=im.shape[:2];th,tw=thumb.height,thumb.width
            fine=torch.einsum('qc,qhw->chw',class_probs[b],full[b,:,:th,:tw]).clamp_min(0)
            fine=F.interpolate(fine[None],size=(h,w),mode='bilinear',align_corners=False)[0]
            fine_ids=fine.argmax(0).cpu().numpy().astype(np.uint8)
            scores=torch.stack([fine[torch.as_tensor(np.flatnonzero(mapping==c),device=args.device)].sum(0) for c in range(3)])
            all_mass=fine.sum(0).clamp_min(1e-8)
            probs=scores/all_mass
            conf,pred=probs.max(0);conf=conf.cpu().numpy();pred=pred.cpu().numpy().astype(np.uint8)
            dense=pred.copy();dense[np.isin(fine_ids,IGNORE_IDS)]=255
            pred[(conf<.70)|np.isin(fine_ids,IGNORE_IDS)]=255
            if alpha is not None:pred[alpha<255]=255
            privacy=np.isin(fine_ids,PRIVACY_IDS).astype(np.uint8)
            privacy=cv2.dilate(privacy,np.ones((9,9),np.uint8))
            blurred=cv2.GaussianBlur(im,(71,71),0)
            private=np.where(privacy[:,:,None].astype(bool),blurred,im).astype(np.uint8)
            sid=r['image_id'];save_png(out/'images'/f'{sid}.png',private);save_png(out/'masks'/f'{sid}.png',pred)
            save_png(out/'dense_proposals'/f'{sid}.png',dense)
            save_png(out/'teacher_labels'/f'{sid}.png',fine_ids);save_png(out/'confidence'/f'{sid}.png',np.rint(conf*255).clip(0,255).astype(np.uint8))
            save_png(out/'privacy_masks'/f'{sid}.png',privacy*255)
            colored=np.zeros_like(im);valid=pred<3;colored[valid]=PALETTE[pred[valid]];colored[~valid]=[180,0,180]
            Image.fromarray(np.rint(.55*private+.45*colored).astype(np.uint8)).resize((768,round(h*768/w))).save(out/'overlays'/f'{sid}.jpg',quality=90)
            counts=np.bincount(pred.ravel(),minlength=256)
            meta={**r,'image':f'images/{sid}.png','mask':f'masks/{sid}.png','label_status':'automatic_pseudo_label','human_reviewed':False,
                  'annotation_policy_sha256':policy['policy_sha256'],'annotation_source_sha256':policy['resume_code_sha256'],
                  'privacy_reviewed':False,'privacy_fraction':float(privacy.mean()),'valid_fraction':float(valid.mean()),
                  'class_pixels':{str(c):int(counts[c]) for c in [0,1,2,255]},'teacher':'facebook/mask2former-swin-large-mapillary-vistas-semantic',
                  'mean_teacher_confidence':float(conf.mean())}
            # Metadata is the final commit marker for resumable annotation.
            (out/'metadata'/f'{sid}.json').write_text(json.dumps(meta,indent=2));done+=1
        if offset%20==0:print(f'annotated {done}/{len(pending)} at {done/(time.time()-start):.2f} images/s',flush=True)
    print(f'annotation complete, newly processed {done}, elapsed {time.time()-start:.1f}s',flush=True)

if __name__=='__main__':main()
