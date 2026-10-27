import torch.nn as nn
import torch
class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ResBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
    def forward(self,x):
            temp = x
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            x = x + temp
            x = self.relu(x)
            return x


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.res_1=ResBlock(64, 64)
        self.res_2 = ResBlock(64, 64)
        self.res_3 = ResBlock(64, 64)
        self.conv2=nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
    def forward(self, x):
        frist=x
        x=self.relu(self.conv1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        rain=self.conv2(x)
        result=frist-rain
        result=torch.clamp(result,0,1)
        return result

