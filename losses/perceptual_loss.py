import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, new, real):
        f_new = self.features(new)
        f_real = self.features(real)
        return self.l1(f_new, f_real)
