import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

def psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(data_range ** 2 / mse)
#每个batch进行处理
def calculate_metrics(img1, img2, lpips_fn):

    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)#1断开梯度2转回cpu(numpy只能处理cpu数据）3转numpy数组4重新排列维度（ssim默认通道在最后）
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img1_np = np.clip(img1_np, 0, 1)#强制锁定值域为0~1
    img2_np = np.clip(img2_np, 0, 1)

    batch_psnr, batch_ssim = 0, 0
    for i in range(img1_np.shape[0]):
        batch_psnr += psnr(img1_np[i], img2_np[i], data_range=1.0)
        batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, multichannel=True, channel_axis=2, win_size=7)

    avg_psnr = batch_psnr / img1_np.shape[0]
    avg_ssim = batch_ssim / img1_np.shape[0]
#lpips是torch模型，用torch算
    img1_lpips = img1.clone() * 2 - 1#lpips:用神经网络学习人类视觉感知，来衡量两张图片的相似度
    img2_lpips = img2.clone() * 2 - 1#通过乘以 2 减 1把 [0,1] 映射到 [-1,1]
#用Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])能实现同样的范围转换

    with torch.no_grad():
        lpips_score = lpips_fn(img1_lpips, img2_lpips).mean().item()

    return avg_psnr, avg_ssim, lpips_score
#得到的是这个 batch 内每张图的平均指标

def save_checkpoint(state, folder="checkpoints", filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, f"{folder}/{filename}")
#state 是 main.py L116 传进来的字典：{"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
# model.state_dict()：模型所有参数（每层权重 + 偏置）组成的字典，key 是层名如 conv1.weight
# optimizer.state_dict()：优化器的状态（Adam 的动量、步数等）
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#训练中每轮存一张对比图：第一行是有雨图、第二行是模型去雨结果、第三行是真值干净图
def save_some_examples(model, loader, epoch, folder, device):
    import os
    model.eval()
    rainy, clean, filename = next(iter(loader))# iter(loader) 把 DataLoader 变成迭代器，next(...) 取第一个 batch。解包成三个：rainy（有雨图）、clean（真值）、filename（文件名列表，这里没用）
    #为什么只取一个 batch？只想存一张示例图看看效果
    rainy, clean = rainy.to(device), clean.to(device)
    if not os.path.exists(folder):#如果输出文件夹不存在就创建它
        os.makedirs(folder)

    with torch.no_grad():
        derained = model(rainy)

    comparison = torch.cat([rainy, derained, clean], dim=0)#沿第 0 维（batch 维）拼接三批图
    save_image(comparison, f"{folder}/comparison_epoch_{epoch}.png", nrow=rainy.shape[0])

    model.train()