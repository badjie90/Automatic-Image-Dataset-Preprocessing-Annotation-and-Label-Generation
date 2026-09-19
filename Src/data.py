import csv,random
from pathlib import Path
import cv2,numpy as np,torch
cv2.setNumThreads(1)
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset

MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
STD=np.array([.229,.224,.225],np.float32)[:,None,None]

def letterbox(rgb,mask=None,height=384,width=640):
    h,w=rgb.shape[:2];scale=min(height/h,width/w);nh,nw=round(h*scale),round(w*scale)
    top=(height-nh)//2;left=(width-nw)//2
    canvas=np.full((height,width,3),[124,116,104],dtype=np.uint8)
    canvas[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    target=None
    if mask is not None:
        target=np.full((height,width),255,np.uint8)
        target[top:top+nh,left:left+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
    return canvas,target,(top,left,nh,nw)

class LisbonDataset(Dataset):
    def __init__(self,root,split,augment=False):
        self.root=Path(root);self.rows=[r for r in csv.DictReader((self.root/'manifest.csv').open()) if r['split']==split and r['usable']=='True']
        self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        # Cached lossless fixed-size inputs preserve geometry and avoid repeated full PNG decoding.
        im=np.array(Image.open(self.root/'training_cache/images'/f"{r['image_id']}.png").convert('RGB'))
        mask=np.array(Image.open(self.root/'training_cache/masks'/f"{r['image_id']}.png"))
        if self.augment:
            if random.random()<.5:im=im[:,::-1].copy();mask=mask[:,::-1].copy()
            im=Image.fromarray(im)
            for cls,lo,hi in [(ImageEnhance.Brightness,.75,1.25),(ImageEnhance.Contrast,.8,1.2),(ImageEnhance.Color,.8,1.2)]:im=cls(im).enhance(random.uniform(lo,hi))
            im=np.array(im)
            # Joint affine scale/translation. Labels use nearest interpolation and ignore padding.
            scale=random.uniform(.85,1.2);h,w=mask.shape
            mat=cv2.getRotationMatrix2D((w/2,h/2),0,scale);mat[:,2]+=[random.uniform(-.04,.04)*w,random.uniform(-.04,.04)*h]
            im=cv2.warpAffine(im,mat,(w,h),flags=cv2.INTER_LINEAR,borderValue=(124,116,104))
            mask=cv2.warpAffine(mask,mat,(w,h),flags=cv2.INTER_NEAREST,borderValue=255)
        x=(im.transpose(2,0,1).astype(np.float32)/255-MEAN)/STD
        return torch.from_numpy(x.copy()),torch.from_numpy(mask.astype(np.int64)),r['image_id']
