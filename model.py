import torch.nn as nn
from torchsummary import summary
import torch

class ResNet(nn.Module):
    def __init__(self,size):
        super(ResNet,self).__init__()
        
        self.conv=nn.Conv2d(3,32,3,padding=1)
        self.conv1=nn.Conv2d(32,32,3,padding=1)
        self.ReLU=nn.ReLU(inplace=False)
        
        self.down1=nn.Conv2d(32,64,3,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,64,3,padding=1)
        
        self.down2=nn.Conv2d(64,128,3,stride=2,padding=1)
        self.conv3=nn.Conv2d(128,128,3,padding=1)
        
        self.down3=nn.Conv2d(128,256,3,padding=1,stride=2)
        self.conv4=nn.Conv2d(256,256,3,padding=1)
        
        self.down4=nn.Conv2d(256,512,3,padding=1,stride=2)
        self.conv5=nn.Conv2d(512,512,3,padding=1)
        
        self.down5=nn.Conv2d(512,1024,3,padding=1,stride=2)
        self.conv6=nn.Conv2d(1024,1024,3,padding=1)
        
        self.convRes1=nn.Conv2d(32,64,1,stride=2)
        self.convRes2=nn.Conv2d(64,128,1,stride=2)
        self.convRes3=nn.Conv2d(128,256,1,stride=2)
        self.convRes4=nn.Conv2d(256,512,1,stride=2)
        self.convRes5=nn.Conv2d(512,1024,1,stride=2)
        
        self.Linear=nn.Linear(1024*(size>>5)**2,1000)
        self.full=nn.Linear(1000,1000)
        self.out=nn.Linear(1000,42)

    def forward(self,x):
        
        x1=self.conv(x)
        x1=self.ReLU(x1)
        x1=self.conv1(x1)
        x1=self.ReLU(x1)
        x2=self.conv1(x1)
        x2=self.ReLU(x2)
        x2=self.conv1(x2)
        x2=self.ReLU(x2)
        
        x3=self.down1(x2+x1)
        x3=self.ReLU(x3)
        x3=self.conv2(x3)
        x3=self.ReLU(x3)
        x4=self.conv2(x3+self.convRes1(x2))
        x4=self.ReLU(x4)
        x4=self.conv2(x4)
        x4=self.ReLU(x4)
        
        x5=self.down2(x4+x3)
        x5=self.ReLU(x5)
        x5=self.conv3(x5)
        x5=self.ReLU(x5)
        x6=self.conv3(x5+self.convRes2(x4))
        x6=self.ReLU(x6)
        x6=self.conv3(x6)
        x6=self.ReLU(x6)
        
        x7=self.down3(x6+x5)
        x7=self.ReLU(x7)
        x7=self.conv4(x7)
        x7=self.ReLU(x7)
        x8=self.conv4(x7+self.convRes3(x6))
        x8=self.ReLU(x8)
        x8=self.conv4(x8)
        x8=self.ReLU(x8)
        
        x9=self.down4(x7+x8)
        x9=self.ReLU(x9)
        x9=self.conv5(x9)
        x9=self.ReLU(x9)
        x10=self.conv5(x9+self.convRes4(x8))
        x10=self.ReLU(x10)
        x10=self.conv5(x10)
        x10=self.ReLU(x10)

        x11=self.down5(x10+x9)
        x11=self.ReLU(x11)
        x11=self.conv6(x11)
        x11=self.ReLU(x11)
        x12=self.conv6(x11+self.convRes5(x10))
        x12=self.ReLU(x12)
        x12=self.conv6(x12)
        x12=self.ReLU(x12)

        out=self.Linear(torch.flatten(x12,start_dim=1))
        out=self.full(out)
        out=self.out(out)
        
        return(out)
        
    
if __name__ =='__main__':
    device='cuda'
    model=ResNet(224).to(device)
    summary(model,(3,224,224))