"""Run exported ONNX on a local image with the identical letterbox convention."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,onnxruntime as ort
from PIL import Image
from src.data import letterbox

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--image',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    rgb=np.array(Image.open(args.image).convert('RGB'));boxed,_,(top,left,h,w)=letterbox(rgb)
    x=np.ascontiguousarray(boxed.transpose(2,0,1)[None],dtype=np.float32)/255
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    s=ort.InferenceSession(str(args.model),sess_options=opts,providers=['CPUExecutionProvider'])
    logits=s.run(['logits'],{'rgb':x})[0][0,:,top:top+h,left:left+w]
    logits=np.stack([cv2.resize(z,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR) for z in logits])
    mask=logits.argmax(0).astype(np.uint8);args.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(args.output)
    print(json.dumps({'output':str(args.output),'class_ids':{'0':'path','1':'obstacle','2':'background'}}))

if __name__=='__main__':main()
