import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, constant_

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1)
    )

def deconv(in_planes, out_planes, scale_factor=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
        nn.UpsamplingBilinear2d(scale_factor=scale_factor),
        nn.LeakyReLU(0.1)
    )

def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        # self.conv3 = conv(2048, 1024)
        # self.deconv3 = deconv(2048, 1024, scale_factor=1)
        
        self.conv2 = conv(1024, 512) 
        self.deconv2 = deconv(1024, 512)

        self.conv1 = conv(512, 256) 
        self.deconv1 = deconv(512, 256)     
        
        self.conv0 = conv(256, 64) 

        self.softmax = nn.Softmax(1)
        self.pred_mask = predict_mask(64, self.assign_ch)

    def forward(self, xs):
        # x3 = self.deconv3(xs[-1])
        # x2 = torch.cat([xs[-2], x3], dim=1)
        # x2 = self.conv3(x2)

        # x2 = self.deconv2(x2)
        # x1 = torch.cat([F.interpolate(xs[-3], size=x2.shape[2:], mode='bilinear'), x2], dim=1)
        # x1 = self.conv2(x1)
        
        # x1 = self.deconv1(x1)
        # x0 = torch.cat([F.interpolate(xs[-4], size=x1.shape[2:], mode='bilinear'), x1], dim=1)
        # x0 = self.conv1(x0)
        
        # x0 = self.conv0(x0)


        x2 = self.deconv2(xs[-1])
        x1 = torch.cat([F.interpolate(xs[-2], size=x2.shape[2:], mode='bilinear'), x2], dim=1)
        x1 = self.conv2(x1)
        
        x1 = self.deconv1(x1)
        x0 = torch.cat([F.interpolate(xs[-3], size=x1.shape[2:], mode='bilinear'), x1], dim=1)
        x0 = self.conv1(x0)
        
        x0 = self.conv0(x0)


        prob = self.pred_mask(x0)
        return self.softmax(prob)
