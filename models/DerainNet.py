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


class Before_rain(nn.Module):
    def __init__(self):
        super(Before_rain, self).__init__()
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

class ResB_8(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ResB_8, self).__init__()
        self.res_1 = ResBlock(in_channel, out_channel)
        self.res_2 = ResBlock(in_channel, out_channel)
        self.res_3 = ResBlock(in_channel, out_channel)
        self.res_4 = ResBlock(in_channel, out_channel)
        self.res_5 = ResBlock(in_channel, out_channel)
        self.res_6 = ResBlock(in_channel, out_channel)
        self.res_7 = ResBlock(in_channel, out_channel)
        self.res_8 = ResBlock(in_channel, out_channel)
    def forward(self,x):
        x=self.res_1(x)
        x=self.res_2(x)
        x=self.res_3(x)
        x=self.res_4(x)
        x=self.res_5(x)
        x=self.res_6(x)
        x=self.res_7(x)
        x=self.res_8(x)
        return x

class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.Upsample2= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.res_1 = ResB_8(64, 64)
        self.res_2 = ResB_8(64, 64)
        self.res_3 = ResB_8(64, 64)
        self.conv2 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        frist=x
        x = self.relu(self.conv1(x))

        temp_1 = x
        x1 = self.res_1(x)
        x1 = x1 + temp_1


        x2 = self.MaxPool_2(x)
        temp_2 = x2
        x2 = self.res_2(x2)
        x2 = x2 + temp_2
        x2 = self.Upsample2(x2)

        x3 = self.MaxPool_3(x)
        temp_3 = x3
        x3 = self.res_3(x3)
        x3 = x3 + temp_3
        x3 = self.Upsample3(x3)

        x_all=torch.cat((x1,x2,x3),1)
        x_all=self.relu(self.conv2(x_all))
        x_all=self.conv3(x_all)
        result = frist - x_all
        result = torch.clamp(result, 0, 1)
        return result


