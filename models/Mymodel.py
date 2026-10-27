import torch.nn as nn
import torch
import torch.nn.functional as F
#前面两个是我第一次对着prenet写的残余，不在后面改完的里面
#分别复现完2篇原论文的分支小板块，我先初步进行了简单的融合拼接写成一个，就是将原本的res_5板块换成rm,然后再加上lrsc
#不知道这样直接拼的效果如何（直接缝合....)，我让ai给我提了一点建议并再次进行了修改


# Residual Block：2 层 3x3 卷积 + ReLU + 恒等残差
# 参考 PReNet[Ren et al. 2019] 的 f_res====》Mymodel 改用 rm了
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        temp = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + temp
        x = self.relu(x)
        return x

#5 个 ResBlock 串联
class ResB_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResB_5, self).__init__()
        self.res_1 = ResBlock(in_channel, out_channel)
        self.res_2 = ResBlock(in_channel, out_channel)
        self.res_3 = ResBlock(in_channel, out_channel)
        self.res_4 = ResBlock(in_channel, out_channel)
        self.res_5 = ResBlock(in_channel, out_channel)

    def forward(self, x):
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        return x

#单层卷积 LSTM，PReNet 的 f_recurrent
#输入 x_t 与上一阶段隐状态 h_{t-1} 沿通道拼接后，一个 3x3 卷积同时算出 4 组门控
#(输入门 i / 遗忘门 f / 输出门 o / 候选记忆 g)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,kernel_size=kernel_size, padding=padding)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

#CA Unit：通道注意力
#MPRNet TFAM 通道分支：全局平均池化=》两组 1x1 卷积各处理一半通道
class ChannelUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_a = nn.Conv2d(half, half, 1)
        self.conv_b = nn.Conv2d(channels - half, channels - half, 1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        stat = self.avg_pool(x)
        b_, c, _, _ = stat.shape
        half = c // 2
        a = self.conv_a(self.act(stat[:, :half]))
        b = self.conv_b(self.act(stat[:, half:]))
        return torch.sigmoid(torch.cat([a, b], dim=1))

#Pos Unit：空间注意力
#MPRNet 的 TFAM 空间分支：这个原本我是按原文复现的，但是ai帮我又改了一下
#沿通道聚合 avg/max -> 压到低分辨率-> 双线性上采样回原尺寸 -> sigmoid
class PositionUnit(nn.Module):
    def __init__(self, channels, k=7, pool_ratio=8):
        super().__init__()
        self.k = k
        self.pool_ratio = pool_ratio
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = max(1, h // self.pool_ratio), max(1, w // self.pool_ratio)
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        desc = F.adaptive_avg_pool2d(desc, (ph, pw))
        att = self.conv(desc)
        att = F.interpolate(att, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(att)

#TFAM：CA(通道) + Pos(空间) 双注意力
#通道权重、空间权重各自乘回特征 -> concat -> 1x1 融合 -> sigmoid 生成掩码 -> 残差相加
#近恒等初始化：初始 mask≈sigmoid(-4)≈0.018，模块近似恒等，训练更稳（ai帮我优化的这个是）
class TwoFoldAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelUnit(channels)
        self.pos = PositionUnit(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.constant_(self.fuse.bias, -4.0)

    def forward(self, x):
        x_ca = x * self.ca(x)
        x_pa = x * self.pos(x)
        mask = torch.sigmoid(self.fuse(torch.cat([x_ca, x_pa], dim=1)))
        return x + x * mask

#ARB：BN(瓶颈) / Res(恒等直连) / Adp(自适应)
class ARB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # BN 路径：Dw(3x3, depthwise) -> BN -> ReLU6 -> Pw(1x1) -> TFAM -> Dw -> BN -> ReLU6 -> Pw
        self.dw1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.tfam = TwoFoldAttention(channels)
        self.dw2 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.ReLU6(inplace=True)

        # Adp 路径：GAP -> 1x1，可学习缩放 delta
        self.adp_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.delta = nn.Parameter(torch.zeros(1))

        # 聚合：恒等Res + BN 之后补一个 Dw，再融合 Adp，最后 1x1
        self.dw_agg = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.pw_agg = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.act(self.bn1(self.dw1(x)))                         # BN
        h = self.pw1(h)              # Pw
        h = self.tfam(h)                              # 注意力
        h = self.act(self.bn2(self.dw2(h)))
        h = self.pw2(h)

        h = h + x
        h = self.dw_agg(h)                                          # 聚合前补 Dw
        h = h + self.delta * self.adp_path(x)                       # Adp
        return self.pw_agg(h)                                       # 融合


#RCB：Residual Concatenation Block，MPRNet
#n个 ARB 逐级输出全部 concat -> 1x1 融合 -> 恒等直连
#拼接后1x1 压缩到原通道
class RCB(nn.Module):
    def __init__(self, channels, num_arb=3):
        super().__init__()
        self.arbs = nn.ModuleList([ARB(channels) for _ in range(num_arb)])
        self.conv = nn.Conv2d(channels * (num_arb + 1), channels, 1)

    def forward(self, x):
        outs = [x]
        for arb in self.arbs:
            x = arb(x)
            outs.append(x)
        return self.conv(torch.cat(outs, dim=1)) + x

#RM：Residual Module，MPRNet
#替换 PReNet  f_res(5 个 ResBlock) 的特征提取器
class ResidualModule(nn.Module):
    def __init__(self, channels, num_rcb=3, num_arb=3):
        super().__init__()
        self.rcbs = nn.ModuleList([RCB(channels, num_arb) for _ in range(num_rcb)])
        self.conv = nn.Conv2d(channels * (num_rcb + 1), channels, 1)

    def forward(self, x):
        outs = [x]
        for rcb in self.rcbs:
            x = rcb(x)
            outs.append(x)
        return self.conv(torch.cat(outs, dim=1)) + x

#融合模型：PReNet 的渐进循环主干 + MPRNet 的特征提取块
#主干still PReNet阶段递归 + ConvLSTM 跨阶段记忆
#特征提取把原 f_res(5 个 ResBlock) 换成 MPRNet 的 RM,再加上lrsc
#参考: PReNet与 MPRNet
class Mymodel(nn.Module):
    def __init__(self, T=6):
        super(Mymodel, self).__init__()
        self.T = T
        self.f_in = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_lstm_cell = ConvLSTMCell(input_dim=64, hidden_dim=64)
        self.rm = ResidualModule(channels=64)
        self.tfam = TwoFoldAttention(channels=64)

        self.f_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        nn.init.zeros_(self.f_out.weight)
        nn.init.zeros_(self.f_out.bias)

    def forward(self, y):
        B, C, H, W = y.shape
        h_state = torch.zeros(B, 64, H, W, device=y.device)
        c_state = torch.zeros(B, 64, H, W, device=y.device)
        x_prev = y
        for _ in range(self.T):
            x_cat = torch.cat([x_prev, y], dim=1)
            x_in = self.f_in(x_cat)
            h_state, c_state = self.conv_lstm_cell(x_in, (h_state, c_state))
            res_feat = self.rm(h_state) + h_state
            res_feat = self.tfam(res_feat)
            rain_layer = self.f_out(res_feat)
            x_prev = torch.clamp(y - rain_layer, 0, 1)
        return x_prev


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
