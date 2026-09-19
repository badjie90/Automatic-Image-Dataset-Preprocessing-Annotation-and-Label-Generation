"""Three-class GCNet-S, CVPR 2025, with the official auxiliary segmentation head."""
import torch
from torch import nn
from torch.nn import functional as F
from .gc_backbone import GCNet
from .gc_compat import ConvModule

class Segmenter(nn.Module):
    def __init__(self,deploy=False):
        super().__init__();self.backbone=GCNet(deploy=deploy)
        def head(i):return nn.Sequential(nn.BatchNorm2d(i),nn.ReLU(inplace=True),ConvModule(i,64,3,padding=1,norm_cfg={'type':'BN'},act_cfg={'type':'ReLU','inplace':True}),nn.Conv2d(64,3,1))
        self.head=head(128);self.aux=nn.Identity() if deploy else head(64)
    def forward(self,x):
        features=self.backbone(x)
        if self.training:
            a,b=features
            return F.interpolate(self.head(b),size=x.shape[-2:],mode='bilinear',align_corners=False),F.interpolate(self.aux(a),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return F.interpolate(self.head(features),size=x.shape[-2:],mode='bilinear',align_corners=False)
    def switch_to_deploy(self):
        self.eval();self.backbone.switch_to_deploy();self.aux=nn.Identity();return self

class DeploymentModel(nn.Module):
    """Accept RGB float32 [0,1] BCHW; return temperature-scaled logits."""
    def __init__(self,model,temperature=1.):
        super().__init__();self.model=model
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.register_buffer('temperature',torch.tensor(float(temperature)))
    def forward(self,rgb):return self.model((rgb-self.mean)/self.std)/self.temperature
