import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #偏置参数会被归一化操作 “抵消” 掉，留着偏置相当于多了无用的参数
            nn.InstanceNorm2d(out_channels), 
            nn.ReLU(inplace=True),#直接在输入张量上修改数据,节省一点显存
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x): 
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(UNet, self).__init__()
        #下采样的目的是通过卷积提取更抽象的深层特征，同时缩小图片尺寸减少计算量
        self.downs = nn.ModuleList()#按顺序存放 3 个 Block：[Block(3→64), Block(64→128), Block(128→256)]
        #上采样的目的是把深层特征图的尺寸放大，同时通过 “跳跃连接” 拼接下采样时保存的浅层高分辨率特征，恢复图片细节
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#1/2

        for feature in features:
            self.downs.append(Block(in_channels, feature))
            in_channels = feature
        #瓶颈层（Bottleneck）,特征最深、通道翻倍
        self.bottleneck = Block(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(Block(feature * 2, feature))
            # self.ups 包含 6 个元素：[ConvTranspose2d(512→256), Block(512→256), ConvTranspose2d(256→128), Block(256→128), ConvTranspose2d(128→64), Block(128→64)]
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        s = x
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)#保存这一层特征，后面跳跃连接
            x = self.pool(x)# 最大池化，尺寸减半，进入下一层，压缩空间信息，突出关键特征

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]#反转
        #跳跃连接：把下采样时的特征图‘抄近路’接到上采样阶段”，下采样的浅特征图保留了清晰的图像细节，上采样的深特征图有全局雨滴信息，拼接后网络能同时用细节和全局信息去雨

        for idx in range(0, len(self.ups), 2):
            #self.ups里面成对存放：[转置卷积, Block, 转置卷积, Block, 转置卷积, Block]所以idx依次取：0, 2, 4，一次处理一对（转置卷积 + Block）
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)#ups[idx+1]就是那一对里的 Block


        return self.final_conv(x) + s
