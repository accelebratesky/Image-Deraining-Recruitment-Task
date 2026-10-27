import torch.nn as nn
import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

class EasyContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.gap=nn.AdaptiveAvgPool2d(1)

    # 这一部分是infoNCE前第一次改造时写的
    '''
    def forward(self, old, new, real):
        f_old = self.features(old)
        old_gap = self.gap(f_old).flatten(1)
        f_new = self.features(new)
        new_gap = self.gap(f_new).flatten(1)
        f_real = self.features(real)
        real_gap = self.gap(f_real).flatten(1)

        dianji_zheng = torch.sum(new_gap * real_gap, dim=1)  
        len_n = torch.norm(new_gap, dim=1)                   
        len_l = torch.norm(real_gap, dim=1)
        cos_1 = dianji_zheng / (len_n * len_l + 1e-8)         

        dianji_fu = torch.sum(new_gap * old_gap, dim=1)      
        len_o = torch.norm(old_gap, dim=1)
        cos_2 = dianji_fu / (len_n * len_o + 1e-8)           

        return -cos_1.mean() + cos_2.mean()
    '''
    #最后一道改后
    def forward(self, old,new, real):
        f_old = self.features(old)#变成（B,512,H,W)
        old_gap = F.normalize(self.gap(f_old).flatten(1))#变成（B，512）

        f_new = self.features(new)
        new_gap=F.normalize(self.gap(f_new).flatten(1))

        B = new_gap.shape[0]#batch数目

        f_real = self.features(real)
        real_gap = F.normalize(self.gap(f_real).flatten(1))

        pos_sim = torch.mm(new_gap,real_gap .t()) / 0.07 #(B,B)!!
        neg_sim = torch.mm(new_gap,old_gap.t()) / 0.07

        logits = torch.cat([pos_sim, neg_sim], dim=1)#（B,2B)

        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)
