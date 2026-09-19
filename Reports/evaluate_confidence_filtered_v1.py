"""Native-resolution test metrics, independent calibration subset, plots, ONNX export."""
import argparse,csv,json,sys,time,copy,hashlib,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'reports/.mplconfig'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import Segmenter,DeploymentModel
from src.data import letterbox,MEAN,STD
from src.metrics import Metrics,NAMES,ratio

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';RUN=ROOT/'runs/gcnet_s'
COLORS=np.array([[0,190,60],[230,45,45],[70,90,130]],np.uint8)

def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields or list(rows[0]));w.writeheader();w.writerows(rows)

@torch.inference_mode()
def predict(model,row,device):
    rgb=np.array(Image.open(DATA/row['image']).convert('RGB'));target=np.array(Image.open(DATA/row['mask']))
    boxed,_,(top,left,h,w)=letterbox(rgb)
    x=torch.from_numpy(((boxed.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD).copy())[None].to(device)
    # Evaluate in float32, matching ONNX export precision.
    logits=model(x)[:,:,top:top+h,left:left+w]
    logits=F.interpolate(logits,size=target.shape,mode='bilinear',align_corners=False)[0]
    return logits,rgb,target

def calibration(model,rows,device):
    rng=np.random.default_rng(20260918);xs=[];ys=[]
    for i,r in enumerate(rows):
        logits,_,target=predict(model,r,device);valid=np.flatnonzero(target.ravel()<3)
        ids=rng.choice(valid,min(4096,len(valid)),replace=False)
        xs.append(logits.flatten(1)[:,torch.as_tensor(ids,device=device)].T.cpu());ys.append(torch.from_numpy(target.ravel()[ids].astype(np.int64)))
    x=torch.cat(xs).float();y=torch.cat(ys);logt=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([logt],lr=.1,max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.cross_entropy(x/logt.exp().clamp(.05,10),y);loss.backward();return loss
    before=float(F.cross_entropy(x,y));opt.step(closure);t=float(logt.detach().exp().clamp(.05,10));after=float(F.cross_entropy(x/t,y))
    if after>before+1e-6:t=1.;after=before
    return t,{'temperature':t,'calibration_images':len(rows),'sampled_pixels':len(y),'sampling':'up to 4096 uniformly sampled valid pixels per image, seed 20260918',
              'reference_type':'automatic_pseudo_labels','nll_before':before,'nll_after':after,'test_used_for_temperature_fit':False}

def make_plots(out,metrics,uncal,history,summary):
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    # Confusion matrix normalised by reference class.
    fig,ax=plt.subplots(figsize=(5,4));cm=ratio(metrics.cm,metrics.cm.sum(1,keepdims=True));im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cm[i,j]:.3f}',ha='center',va='center',color='white' if cm[i,j]>.5 else 'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=NAMES,yticklabels=NAMES,xlabel='Prediction',ylabel='Automatic reference',title='Test confusion matrix');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5));ax.bar(NAMES,[summary[f'IoU_{n}'] for n in NAMES],color=['#00a640','#cf3333','#465a82']);ax.set(ylim=(0,1),ylabel='IoU',title='Agreement with automatic test labels');fig.tight_layout();fig.savefig(out/'class_iou.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    bins=[]
    for m,label in [(uncal,'Before temperature scaling'),(metrics,'After temperature scaling')]:
        c=ratio(m.conf,m.count);acc=ratio(m.correct,m.count);axs[0].plot(c,acc,'o-',label=label)
    axs[0].plot([0,1],[0,1],'k--');axs[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Agreement with automatic labels',title='Reliability diagram');axs[0].legend(fontsize=8)
    axs[1].bar((np.arange(metrics.bins)+.5)/metrics.bins,metrics.count/max(metrics.n,1),width=.9/metrics.bins);axs[1].set(xlabel='Confidence',ylabel='Fraction of evaluated pixels',title='Confidence distribution')
    fig.tight_layout();fig.savefig(out/'calibration.png');plt.close(fig)
    for i in range(metrics.bins):bins.append({'bin_lower':i/metrics.bins,'bin_upper':(i+1)/metrics.bins,'count':int(metrics.count[i]),'mean_confidence':float(ratio(metrics.conf[i],metrics.count[i])),'accuracy':float(ratio(metrics.correct[i],metrics.count[i]))})
    write_csv(out/'reliability_bins.csv',bins)
    tp=metrics.path_hist[0,::-1].cumsum()[::-1];fp=metrics.path_hist[1,::-1].cumsum()[::-1]
    precision=ratio(tp,tp+fp);recall=ratio(tp,metrics.path_hist[0].sum())
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(recall,precision);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall of path pixels',ylabel='Precision of path pixels',title='Path probability threshold sweep');fig.tight_layout();fig.savefig(out/'path_precision_recall.png');plt.close(fig)
    write_csv(out/'path_precision_recall.csv',[{'threshold':i/1000,'precision':float(precision[i]),'recall':float(recall[i])} for i in range(1000)])
    correct=metrics.risk_hist[0,::-1].cumsum();errors=metrics.risk_hist[1,::-1].cumsum();coverage=(correct+errors)/max(metrics.n,1);risk=ratio(errors,correct+errors)
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(coverage,risk);ax.set(xlabel='Coverage of evaluated pixels',ylabel='Pixel error rate',title='Confidence-based selective prediction');fig.tight_layout();fig.savefig(out/'risk_coverage.png');plt.close(fig)
    write_csv(out/'risk_coverage.csv',[{'coverage':float(c),'risk':float(r)} for c,r in zip(coverage,risk)])
    epochs=[int(r['epoch']) for r in history];fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    axs[0].plot(epochs,[float(r['train_loss']) for r in history],label='Training objective');axs[0].plot(epochs,[float(r['val_nll']) for r in history],label='Validation NLL');axs[0].set(xlabel='Epoch',ylabel='Loss');axs[0].legend()
    axs[1].plot(epochs,[float(r['val_mIoU']) for r in history],label='mIoU');axs[1].plot(epochs,[float(r['val_IoU_path']) for r in history],label='Path IoU');axs[1].set(xlabel='Epoch',ylabel='Automatic-label agreement');axs[1].legend();fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)

def export_model(model,temperature,rows,device):
    import onnx,onnxruntime as ort
    folder=RUN/'export';folder.mkdir(exist_ok=True)
    original=DeploymentModel(copy.deepcopy(model).cpu().eval(),temperature)
    fused=DeploymentModel(copy.deepcopy(model).cpu().eval().switch_to_deploy(),temperature).eval()
    samples=[]
    for row in rows[:3]:
        rgb=np.array(Image.open(DATA/row['image']).convert('RGB'));boxed,_,_=letterbox(rgb)
        samples.append(torch.from_numpy(boxed.transpose(2,0,1).astype(np.float32)/255)[None])
    maximum=0.;relative=0.
    with torch.inference_mode():
        for x in samples:
            a=original(x);b=fused(x);maximum=max(maximum,float((a-b).abs().max()));relative=max(relative,float((a-b).abs().max()/a.abs().max().clamp_min(1e-8)))
            assert torch.allclose(a,b,atol=.002,rtol=.0002),'GCNet block fusion failed numerical verification'
    path=folder/'gcnet_s_lisbon.onnx'
    torch.onnx.export(fused,samples[0],str(path),input_names=['rgb'],output_names=['logits'],opset_version=13,dynamo=False,
                      do_constant_folding=True)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    onnx_error=0.;agreement=[]
    with torch.inference_mode():
        for x in samples:
            expected=fused(x).numpy();actual=session.run(None,{'rgb':x.numpy()})[0]
            onnx_error=max(onnx_error,float(np.abs(expected-actual).max()));agreement.append(float((expected.argmax(1)==actual.argmax(1)).mean()))
            np.testing.assert_allclose(actual,expected,atol=.002,rtol=.0002)
    torch.save({'model':fused.model.state_dict(),'temperature':temperature,'deploy':True,'architecture':'GCNet-S'},folder/'gcnet_s_lisbon_deploy.pt')
    # Batch-one GPU benchmark on the available server, not on a Jetson Nano.
    fused=fused.to(device);x=samples[0].to(device);times=[]
    with torch.inference_mode():
        for _ in range(10):fused(x)
        torch.cuda.synchronize(device)
        for _ in range(50):
            start=time.perf_counter();fused(x);torch.cuda.synchronize(device);times.append((time.perf_counter()-start)*1000)
    report={'onnx_file':str(path.relative_to(ROOT)),'opset':13,'input_name':'rgb','input_shape':[1,3,384,640],'input_dtype':'float32',
            'input_range':[0,1],'input_colour':'RGB','normalisation':'included in ONNX graph','temperature_scaling':'included in ONNX graph',
            'output_name':'logits','output_shape':[1,3,384,640],'output_order':NAMES,'output_postprocess':'crop letterbox padding, resize logits bilinearly to original image, then softmax and argmax',
            'mask_ignore_id':255,'fusion_max_abs_error':maximum,'fusion_relative_max_error':relative,'onnxruntime_max_abs_error':onnx_error,
            'onnxruntime_label_agreement':agreement,'onnx_checker':'passed','parity_images':len(samples),
            'deployment_parameters':sum(p.numel() for p in fused.model.parameters()),'onnx_bytes':path.stat().st_size,
            'gpu_name':torch.cuda.get_device_name(device),'gpu_fp32_median_ms':float(np.median(times)),'gpu_fp32_p95_ms':float(np.percentile(times,95)),
            'benchmark_scope':'model only, batch 1, 10 warmup and 50 measured iterations; excludes disk, preprocessing, and postprocessing',
            'jetson_nano_tested':False,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder/'deployment.json').write_text(json.dumps(report,indent=2));print('ONNX verified',json.dumps(report),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');args=ap.parse_args();torch.set_num_threads(4)
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():ap.error('This experiment driver requires CUDA for evaluation and the GPU benchmark. infer_onnx.py supports CPU inference.')
    out=RUN/'evaluation';out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in csv.DictReader((DATA/'manifest.csv').open()) if r['usable']=='True']
    test=[r for r in rows if r['split']=='test'];cal=[r for r in rows if r['split']=='calibration'];val=[r for r in rows if r['split']=='val']
    ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False);model=Segmenter().to(args.device).eval();model.load_state_dict(ck['model'])
    temp,calstats=calibration(model,cal,args.device);(out/'temperature.json').write_text(json.dumps(calstats,indent=2));print('temperature',temp,flush=True)
    before=Metrics();after=Metrics();per_image=[];start=time.time();full_pixels=0
    qualitative=set(np.linspace(0,len(test)-1,min(12,len(test))).astype(int))
    for i,row in enumerate(test):
        logits,rgb,target=predict(model,row,args.device);p0=logits.softmax(0).cpu().numpy();p=(logits/temp).softmax(0).cpu().numpy()
        before.update(p0,target,with_boundary=False);after.update(p,target);one=Metrics();one.update(p,target,with_boundary=False)
        per_image.append({'image_id':row['image_id'],'reference_type':'automatic_pseudo_labels','valid_pixel_fraction':float((target<3).mean()),**{k:v for k,v in one.summary().items() if not k.startswith('BF1')}});full_pixels+=target.size
        if i in qualitative:
            pred=p.argmax(0);reference=np.full_like(rgb,[180,0,180]);v=target<3;reference[v]=COLORS[target[v]]
            fig,axs=plt.subplots(1,4,figsize=(16,3));axs[0].imshow(rgb);axs[0].set_title('Privacy-processed input')
            axs[1].imshow(reference);axs[1].set_title('Automatic reference')
            axs[2].imshow(COLORS[pred]);axs[2].set_title('GCNet-S prediction')
            axs[3].imshow(np.where(v,pred!=target,np.nan),vmin=0,vmax=1,cmap='Reds');axs[3].set_title('Disagreement on valid pixels')
            for ax in axs:ax.axis('off')
            fig.suptitle(row['image_id']);fig.tight_layout();fig.savefig(out/f'qualitative_{row["image_id"]}.png',dpi=120);plt.close(fig)
        if i%25==0:print(f'evaluated {i+1}/{len(test)} in {time.time()-start:.1f}s',flush=True)
    assert np.array_equal(before.cm,after.cm),'Positive temperature must preserve argmax labels'
    common={'model':'GCNet-S','checkpoint_epoch':ck['epoch']+1,'split':'test','reference_type':'automatic_pseudo_labels','route_independence_verified':False,
            'human_reviewed':False,'test_images':len(test),'evaluated_pixel_fraction':after.n/full_pixels,'ECE_bins':15,'BF1_tolerance_pixels':3,
            'evaluation_resolution':'native image resolution, restored logits','Brier_definition':'mean sum over 3 classes of squared probability error; range 0 to 2'}
    calibrated={**common,'temperature_scaled':True,'temperature':temp,**after.summary()}
    raw={**common,'temperature_scaled':False,'temperature':1.,**before.summary()}
    # Temperature does not change argmax. Reuse the boundary counts measured once.
    for key,value in after.summary().items():
        if key.startswith('BF1'):raw[key]=value
    write_csv(out/'metrics.csv',[raw,calibrated]);write_csv(out/'per_image_metrics.csv',per_image)
    np.savetxt(out/'confusion_counts.csv',after.cm,delimiter=',',fmt='%d',header=','.join(NAMES),comments='')
    history=list(csv.DictReader((RUN/'history.csv').open()));make_plots(out,after,before,history,calibrated)
    (out/'protocol.json').write_text(json.dumps({**common,'ECE':'top-label confidence, 15 equal-width bins, global pixel weighting, confidence 1 included in last bin',
        'BF1':'classwise inner boundaries from four-connected erosion, Euclidean distance tolerance 3 native pixels; exclude ignore/image-border neighbourhood using four iterations of four-connected erosion; global matched counts then macro average; both boundary sets empty gives NaN and one empty set gives zero',
        'IoU':'global confusion-matrix class IoU; mean over classes with nonzero union',
        'path_precision_recall':'argmax confusion matrix for CSV scalar; threshold sweep plot uses path probability',
        'caveat':'agreement and calibration relative to automatically generated labels, conditioned on the teacher confidence filter; not verified real-world accuracy or safety',
        'test_used_for_model_selection':False,'test_used_for_temperature_fit':False,'calibration':calstats},indent=2))
    export_model(model,temp,val,args.device)
    (RUN/'evaluation_complete.json').write_text(json.dumps(calibrated,indent=2));print(json.dumps(calibrated,indent=2),flush=True)

if __name__=='__main__':main()
