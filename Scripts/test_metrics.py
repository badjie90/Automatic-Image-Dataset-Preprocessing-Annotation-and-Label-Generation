"""Analytical metric cases and ignore-boundary behaviour."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.metrics import Metrics

def main():
    y=np.zeros((32,32),dtype=np.uint8);y[:,16:]=1;y[20:]=2
    p=np.eye(3,dtype=np.float32)[y].transpose(2,0,1)
    m=Metrics();m.update(p,y);s=m.summary()
    for key in ['mIoU','Precision_path','Recall_path','BF1_macro']:assert abs(s[key]-1)<1e-7,(key,s)
    assert abs(s['ECE'])<1e-7 and abs(s['Brier_score'])<1e-7
    wrong=p[[2,0,1]];m=Metrics();m.update(wrong,y);s=m.summary()
    assert abs(s['mIoU'])<1e-7 and abs(s['ECE']-1)<1e-7 and abs(s['Brier_score']-2)<1e-7,s
    q=np.full_like(p,1/3);m=Metrics();m.update(q,y);assert abs(m.summary()['Brier_score']-2/3)<1e-6
    yy=y.copy();yy[0:5]=255;m=Metrics();m.update(p,yy);assert m.n==int((yy<3).sum()) and abs(m.summary()['mIoU']-1)<1e-7
    # Streaming aggregation equals a single aggregate for non-boundary metrics.
    a=Metrics();a.update(q[:,:,:16],y[:,:16],False);a.update(q[:,:,16:],y[:,16:],False)
    b=Metrics();b.update(q,y,False)
    for k in ['mIoU','ECE','Brier_score','NLL']:assert np.isclose(a.summary()[k],b.summary()[k]),k
    # A one-pixel shift remains within the documented three-pixel boundary tolerance.
    shifted=np.roll(p,1,axis=2);m=Metrics();m.update(shifted,y);assert m.summary()['BF1_macro']>.95
    print('PASS perfect, wrong, uniform, ignored, streaming, and boundary-tolerance cases')

if __name__=='__main__':main()
