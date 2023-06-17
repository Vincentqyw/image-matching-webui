import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

class ResNet18(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.net = tvm.resnet18(pretrained=pretrained)
    def forward(self, x):
        self = self.net
        x1 = x
        x = self.conv1(x1)
        x = self.bn1(x)
        x2 = self.relu(x)
        x = self.maxpool(x2)
        x4 = self.layer1(x)
        x8 = self.layer2(x4)
        x16 = self.layer3(x8)
        x32 = self.layer4(x16)
        return {32:x32,16:x16,8:x8,4:x4,2:x2,1:x1}

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, dilation = None, freeze_bn = True, anti_aliased = False) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
    def forward(self, x):
        net = self.net
        feats = {1:x}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x 
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x 
        x = net.layer2(x)
        feats[8] = x  
        x = net.layer3(x)
        feats[16] = x
        x = net.layer4(x)
        feats[32] = x
        return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass




class ResNet101(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None) -> None:
        super().__init__()
        if weights is not None:
            self.net = tvm.resnet101(weights = weights)
        else:
            self.net = tvm.resnet101(pretrained=pretrained)
        self.high_res = high_res
        self.scale_factor = 1 if not high_res else 1.5
    def forward(self, x):
        net = self.net
        feats = {1:x}
        sf = self.scale_factor
        if self.high_res:
            x = F.interpolate(x, scale_factor=sf, align_corners=False, mode="bicubic")
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer2(x)
        feats[8] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer3(x)
        feats[16] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer4(x)
        feats[32] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        return feats

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass


class WideResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None) -> None:
        super().__init__()
        if weights is not None:
            self.net = tvm.wide_resnet50_2(weights = weights)
        else:
            self.net = tvm.wide_resnet50_2(pretrained=pretrained)
        self.high_res = high_res
        self.scale_factor = 1 if not high_res else 1.5
    def forward(self, x):
        net = self.net
        feats = {1:x}
        sf = self.scale_factor
        if self.high_res:
            x = F.interpolate(x, scale_factor=sf, align_corners=False, mode="bicubic")
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer2(x)
        feats[8] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer3(x)
        feats[16] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        x = net.layer4(x)
        feats[32] = x if not self.high_res else F.interpolate(x,scale_factor=1/sf,align_corners=False, mode="bilinear")    
        return feats

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass