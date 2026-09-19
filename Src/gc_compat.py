"""Minimal PyTorch equivalents of the MMCV primitives used by the official GCNet-S.

Only the BN/ReLU configuration of the original small model is supported.
"""
import torch
from torch import nn
from torch.nn import functional as F

class BaseModule(nn.Module):
    def __init__(self,init_cfg=None):super().__init__()

def build_norm_layer(cfg,num_features):
    assert cfg.get('type','BN')=='BN'
    return 'bn',nn.BatchNorm2d(num_features,eps=cfg.get('eps',1e-5),momentum=cfg.get('momentum',.1))

def build_activation_layer(cfg):
    assert cfg['type']=='ReLU'
    return nn.ReLU(inplace=cfg.get('inplace',True))

class ConvModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias='auto',norm_cfg=None,act_cfg=None,order=('conv','norm','act')):
        super().__init__();self.order=order
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=(norm_cfg is None) if bias=='auto' else bias)
        self.bn=build_norm_layer(norm_cfg,in_channels if order.index('norm')<order.index('conv') else out_channels)[1] if norm_cfg else nn.Identity()
        self.activate=build_activation_layer(act_cfg) if act_cfg else nn.Identity()
    def forward(self,x):
        for key in self.order:x={'conv':self.conv,'norm':self.bn,'act':self.activate}[key](x)
        return x

def resize(x,size,mode='bilinear',align_corners=False):return F.interpolate(x,size=size,mode=mode,align_corners=align_corners)

class DAPPM(BaseModule):
    """DAPPM architecture copied from the upstream GCNet OpenMMLab dependency."""
    def __init__(self,in_channels,branch_channels,out_channels,num_scales=5,norm_cfg=None,act_cfg=None):
        super().__init__();assert num_scales==5
        def conv(i,o,k=1):return ConvModule(i,o,k,padding=k//2,norm_cfg=norm_cfg,act_cfg=act_cfg,order=('norm','act','conv'),bias=False)
        self.scales=nn.ModuleList([conv(in_channels,branch_channels)])
        for k,s,p in [(5,2,2),(9,4,4),(17,8,8)]:self.scales.append(nn.Sequential(nn.AvgPool2d(k,s,p),conv(in_channels,branch_channels)))
        self.scales.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),conv(in_channels,branch_channels)))
        self.processes=nn.ModuleList([conv(branch_channels,branch_channels,3) for _ in range(4)])
        self.compression=conv(branch_channels*5,out_channels);self.shortcut=conv(in_channels,out_channels)
    def forward(self,x):
        fs=[self.scales[0](x)]
        for i in range(1,5):fs.append(self.processes[i-1](F.interpolate(self.scales[i](x),size=x.shape[2:],mode='bilinear',align_corners=False)+fs[-1]))
        return self.compression(torch.cat(fs,1))+self.shortcut(x)
