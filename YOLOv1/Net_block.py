import torch
import torch.nn as nn

class ResNet_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64),
        self.relu=nn.ReLU(inplace=True),
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1_1=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1_1=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2_1=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2_2=nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3_1=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer3_2=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4_1=nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer4_2=nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.conv1(x)#[BC,H/2,W/2]
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)#[BC,H/4,W/4]
        x=self.layer1_1(x)+x
        x=self.layer1_2(x)+x#[BC,H/4,W/4]
        x=self.layer2_1(x)+x
        x=self.layer2_2(x)+x#[BC,H/8,W/8]
        x=self.layer3_1(x)+x
        x=self.layer3_2(x)+x#[BC,H/16,W/16]
        x=self.layer4_1(x)+x
        x=self.layer4_2(x)+x#[BC,H/32,W/32]
        return x

class SPP(nn.Module):
    def __init__(self,in_channel=512):
        super().__init__()
        self.cv1=nn.Conv2d(in_channel,in_channel//2,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(in_channel//2)
        self.relu=nn.LeakyReLU(0.1,inplace=True)
        self.m=nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.cv2=nn.Conv2d(in_channel*2,in_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(in_channel)
    def forward(self,x):
        x=self.cv1(x)#[B,C/2,H,W]
        x=self.bn1(x)
        x1=self.relu(x)
        x2=self.m(x1)
        x3=self.m(x2)
        x4=self.m(x3)
        x=torch.cat([x1,x2,x3,x4],dim=1)#[B,C*2,H,W]
        x=self.cv2(x)#[B,C,H,W]
        x=self.bn2(x)
        x=self.relu(x)
        return x#[B,C,H,W]

class DecoupledHead(nn.Module):
    def __init__(self,in_channel=512):
        super().__init__()
        self.cls_conv=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        
        self.reg_conv=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        cls=self.cls_conv(x)
        reg=self.reg_conv(x)
        return cls,reg

