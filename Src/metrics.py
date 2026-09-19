"""Streaming segmentation metrics. Ignore ID 255 never enters a denominator."""
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt

NAMES=['path','obstacle','background']

def ratio(a,b):return np.divide(a,b,out=np.full(np.broadcast(a,b).shape,np.nan,dtype=float),where=np.asarray(b)!=0)

def confusion_metrics(cm):
    cm=np.asarray(cm,dtype=float);tp=np.diag(cm);gt=cm.sum(1);pred=cm.sum(0)
    iou=ratio(tp,gt+pred-tp);p=ratio(tp,pred);r=ratio(tp,gt)
    return {**{f'IoU_{n}':float(v) for n,v in zip(NAMES,iou)},'mIoU':float(np.nanmean(iou)),
            'Precision_path':float(p[0]),'Recall_path':float(r[0]),'Pixel_Accuracy':float(ratio(tp.sum(),cm.sum()))}

class Metrics:
    def __init__(self,bins=15,boundary_tolerance=3):
        self.cm=np.zeros((3,3),np.int64);self.bins=bins;self.tol=boundary_tolerance
        self.count=np.zeros(bins,np.int64);self.conf=np.zeros(bins);self.correct=np.zeros(bins)
        self.brier=0.;self.nll=0.;self.n=0
        self.boundary=np.zeros((3,4),np.int64) # matched prediction, prediction total, matched reference, reference total
        self.path_hist=np.zeros((2,1000),np.int64)
        self.risk_hist=np.zeros((2,1000),np.int64)
    def update(self,probs,target,with_boundary=True):
        probs=np.asarray(probs,dtype=np.float32);target=np.asarray(target)
        assert probs.shape==(3,*target.shape)
        assert np.isfinite(probs).all() and np.allclose(probs.sum(0),1,atol=2e-4)
        valid=target<3;pred=probs.argmax(0);y=target[valid].astype(int);p=probs[:,valid]
        if not y.size:return
        pr=pred[valid];n=y.size;self.n+=n
        self.cm+=np.bincount(3*y+pr,minlength=9).reshape(3,3)
        confid=p.max(0);correct=(pr==y);idx=np.minimum((confid*self.bins).astype(int),self.bins-1)
        self.count+=np.bincount(idx,minlength=self.bins)
        self.conf+=np.bincount(idx,weights=confid,minlength=self.bins)
        self.correct+=np.bincount(idx,weights=correct,minlength=self.bins)
        pt=p[y,np.arange(n)]
        self.brier+=float(np.sum((p*p).sum(0)-2*pt+1,dtype=np.float64))
        self.nll+=float(np.sum(-np.log(np.maximum(pt,1e-12)),dtype=np.float64))
        ix=np.minimum((p[0]*1000).astype(int),999)
        self.path_hist[0]+=np.bincount(ix[y==0],minlength=1000);self.path_hist[1]+=np.bincount(ix[y!=0],minlength=1000)
        ix=np.minimum((confid*1000).astype(int),999)
        self.risk_hist[0]+=np.bincount(ix[correct],minlength=1000);self.risk_hist[1]+=np.bincount(ix[~correct],minlength=1000)
        if with_boundary:
            # Exclude boundaries influenced by ignore regions or by the image frame.
            domain=binary_erosion(valid,iterations=self.tol+1,border_value=0)
            for c in range(3):
                pm=(pred==c)&valid;tm=(target==c)&valid
                pb=pm & ~binary_erosion(pm,border_value=1) & domain
                tb=tm & ~binary_erosion(tm,border_value=1) & domain
                npb=int(pb.sum());ntb=int(tb.sum())
                mp=int((pb & (distance_transform_edt(~tb)<=self.tol)).sum()) if ntb else 0
                mt=int((tb & (distance_transform_edt(~pb)<=self.tol)).sum()) if npb else 0
                self.boundary[c]+=[mp,npb,mt,ntb]
    def summary(self):
        r=confusion_metrics(self.cm)
        gap=np.abs(ratio(self.correct,self.count)-ratio(self.conf,self.count))
        r.update(ECE=float(np.nansum(gap*self.count)/self.n) if self.n else float('nan'),
                 Brier_score=self.brier/self.n if self.n else float('nan'),NLL=self.nll/self.n if self.n else float('nan'),valid_pixels=self.n)
        bfs=[]
        for name,(mp,npb,mt,ntb) in zip(NAMES,self.boundary):
            if npb==0 and ntb==0:f=float('nan')
            elif npb==0 or ntb==0:f=0.
            else:
                p=mp/npb;rcl=mt/ntb;f=2*p*rcl/(p+rcl) if p+rcl else 0.
            r[f'BF1_{name}']=f;bfs.append(f)
        r['BF1_macro']=float(np.nanmean(bfs)) if np.isfinite(bfs).any() else float('nan')
        return r
