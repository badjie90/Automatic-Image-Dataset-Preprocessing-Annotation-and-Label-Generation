"""GCNet-S training and checkpoint selection using validation only."""
import argparse,csv,json,random,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.model import Segmenter
from src.data import LisbonDataset
from src.metrics import confusion_metrics

def loss_fn(logits,y,weights):
    valid=y!=255
    if not valid.any():return logits.sum()*0
    ce=F.cross_entropy(logits,y,weight=weights,ignore_index=255)
    p=logits.float().softmax(1);oh=F.one_hot(y.clamp_max(2),3).permute(0,3,1,2).float()
    v=valid[:,None];num=2*(p*oh*v).sum((0,2,3));den=((p+oh)*v).sum((0,2,3))
    dice=1-((num+1)/(den+1)).mean()
    return ce+.5*dice

@torch.inference_mode()
def validate(model,loader,device):
    model.eval();cm=torch.zeros((3,3),device=device,dtype=torch.long);total=0.;pixels=0
    for x,y,_ in loader:
        x=x.to(device);y=y.to(device)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        pred=logits.argmax(1);v=y!=255
        cm+=torch.bincount(3*y[v]+pred[v],minlength=9).reshape(3,3)
        total+=float(F.cross_entropy(logits.float(),y,ignore_index=255,reduction='sum'));pixels+=int(v.sum())
    return confusion_metrics(cm.cpu().numpy()),total/max(pixels,1)

def seed_worker(_):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=12)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];run=root/'runs/gcnet_s';run.mkdir(parents=True,exist_ok=True)
    seed=20260918;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    train=LisbonDataset(root/'data','train',True);val=LisbonDataset(root/'data','val')
    assert len(train)>0 and len(val)>0
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=6,pin_memory=True,drop_last=True,worker_init_fn=seed_worker,generator=gen,persistent_workers=True)
    vl=DataLoader(val,batch_size=args.batch_size,num_workers=4,pin_memory=True,persistent_workers=True)
    stats=json.loads((root/'reports/dataset_summary.json').read_text());pix=np.array([stats['class_pixels']['train'][str(c)] for c in range(3)],float)
    weights=(pix.sum()/np.maximum(pix,1))**.5;weights=(weights/weights.mean()).clip(.5,3)
    weights=torch.tensor(weights,dtype=torch.float32,device=args.device)
    model=Segmenter().to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');best=-1.;start_epoch=0;stale=0;history=[];step=0
    total_steps=args.epochs*len(loader);warmup=max(len(loader)*2,1)
    config={**vars(args),'architecture':'GCNet-S CVPR 2025','initialisation':'from scratch, official topology with three-class head','seed':seed,
            'class_weights':weights.tolist(),'loss':'weighted cross entropy + 0.5 soft Dice, auxiliary weight 0.4',
            'label_type':'automatic_pseudo_labels','input_height':384,'input_width':640,'validation_resolution':'384x640 letterboxed grid, padding ignored','training_images':len(train),'validation_images':len(val),
            'train_parameters':sum(p.numel() for p in model.parameters()),'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(args.device),
            'manifest_sha256':hashlib.sha256((root/'data/manifest.csv').read_bytes()).hexdigest(),'determinism':'seeded; cuDNN benchmark enabled; bitwise GPU reproducibility not guaranteed'}
    (run/'config.json').write_text(json.dumps(config,indent=2))
    if args.resume and (run/'last.pt').exists():
        ck=torch.load(run/'last.pt',map_location=args.device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);best=ck['best'];step=ck['step'];start_epoch=ck['epoch']+1;history=ck['history'];stale=ck['stale']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu());gen.set_state(ck['loader_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    for epoch in range(start_epoch,args.epochs):
        model.train();begin=time.time();s=0.;n=0
        for x,y,_ in loader:
            lr=args.lr*min(1.,(step+1)/warmup)*max(0.,1-step/total_steps)**.9
            for group in opt.param_groups:group['lr']=lr
            x=x.to(args.device,non_blocking=True);y=y.to(args.device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits,aux=model(x);loss=loss_fn(logits,y,weights)+.4*loss_fn(aux,y,weights)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite training loss at step {step}')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
            s+=float(loss.detach())*len(x);n+=len(x);step+=1
        vm,vce=validate(model,vl,args.device);row={'epoch':epoch+1,'step':step,'train_loss':s/n,'val_nll':vce,'val_mIoU':vm['mIoU'],'val_IoU_path':vm['IoU_path'],'lr':lr,'seconds':time.time()-begin};history.append(row)
        improved=vm['mIoU']>best+1e-4
        if improved:best=vm['mIoU'];stale=0
        else:stale+=1
        payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'best':best,'epoch':epoch,'step':step,'stale':stale,'history':history,'config':config,
                 'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':gen.get_state()}
        torch.save(payload,run/'last.tmp');(run/'last.tmp').replace(run/'last.pt')
        if improved:torch.save({'model':model.state_dict(),'epoch':epoch,'val_mIoU':best,'config':config},run/'best.pt')
        with (run/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(row));w.writeheader();w.writerows(history)
        print(json.dumps(row),flush=True)
        if stale>=args.patience and epoch>=19:print('validation early stopping',flush=True);break
    (run/'training_complete.json').write_text(json.dumps({'completed_epochs':len(history),'best_val_mIoU':best,'stopped_early':len(history)<args.epochs},indent=2))

if __name__=='__main__':main()
