import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import random
import numpy as np
import lpips

from dataset import DerainDataset
from utils import calculate_metrics, save_checkpoint, load_checkpoint, save_some_examples
from models.baseline_net import BaselineNet
from models.unet import UNet
from models.DerainNet import DerainNet
from models.Mymodel import Mymodel

from losses.perceptual_loss import PerceptualLoss
from losses.easy_contrastive_loss import EasyContrastiveLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Deraining Model Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","derainnet","mymodel"], help="Model to use")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Train or test mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing")

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--use_perceptual_loss", action="store_true", help="Use perceptual loss")
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Weight for pixel loss")#像素损失的权重
    parser.add_argument("--lambda_perceptual", type=float, default=0.01, help="Weight for perceptual loss")#感知损失的权重

    parser.add_argument("--use_easy_contrastive", action="store_true", help="Use easy contrastive loss")#是否启用简易对比损失
    parser.add_argument("--lambda_easy_contrastive", type=float, default=0.01, help="Weight for easy contrastive loss")#简易对比损失的权重

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")#随机种子，默认42

    return parser.parse_args()#解析终端输入，打包成 args 对象返回


def train_one_epoch(loader, model, optimizer, pixel_loss_fn, perceptual_loss_fn, easy_contrastive_fn, args, device):
    loop = tqdm(loader, leave=True)#用 tqdm 把加载器包起来，训练时显示进度条，leave表示一直会有显示
    model.train()
#dataloader 每次返回的是一个元组，里面包含三个元素：有雨图、干净图、文件名。这里用 (rainy, clean, _) 是 “解包” 操作，把元组里的前两个元素分别赋值给 rainy 和 clean 变量，第三个文件名用`_`占位忽略
    for _, (rainy, clean, _) in enumerate(loop):#第一个下划线返回的是下标
        rainy, clean = rainy.to(device), clean.to(device)
        derained = model(rainy)
        total_loss = args.lambda_pixel * pixel_loss_fn(derained, clean)#像素损失 = MSE*权重
        postfix = {"pixel_loss": total_loss.item()}#进度条右侧要显示的信息，先放像素损失

        if args.use_perceptual_loss:
            p_loss = args.lambda_perceptual * perceptual_loss_fn(derained, clean)
            total_loss += p_loss
            postfix["perceptual_loss"] = p_loss.item()#把感知损失也加进进度条显示

        if args.use_easy_contrastive:
            c_loss = args.lambda_easy_contrastive * easy_contrastive_fn(rainy, derained, clean)
            total_loss += c_loss
            postfix["contrastive_loss"] = c_loss.item()#把对比损失也加进进度条显示

        loop.set_postfix(**postfix)#统一把上面累积的 postfix 显示在进度条右侧
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def evaluate(loader, model, device, lpips_fn):
    model.eval()
    total_psnr, total_ssim, total_lpips = 0, 0, 0

    with torch.no_grad():
        for rainy, clean, _ in loader:
            rainy, clean = rainy.to(device), clean.to(device)
            derained = model(rainy)

            psnr, ssim, lpips_score = calculate_metrics(derained, clean, lpips_fn)
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score

    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    avg_lpips = total_lpips / len(loader)
#除以的是 len(loader)（batch 数），如果最后一个 batch 不满，得到的不是严格的 "样本平均" 而是 "batch 平均"
    print(f"EVAL => Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f} | Avg LPIPS: {avg_lpips:.4f}")
    return avg_psnr


def main():
    args = get_args()
    set_seed(args.seed)   # 在所有随机操作前固定种子，保证可复现
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model == "baseline":
        model = BaselineNet().to(device)
    elif args.model == "unet":
        model = UNet().to(device)
    elif args.model == "mymodel":
        model = Mymodel().to(device)
    elif args.model == "derainnet":
        model = DerainNet().to(device)


    lpips_fn = lpips.LPIPS(net='alex').to(device)#加载 LPIPS 指标网络（AlexNet 后端）

    pixel_loss_fn = nn.MSELoss()#像素损失用 MSE
    perceptual_loss_fn = PerceptualLoss().to(device) if args.use_perceptual_loss else None
    easy_contrastive_fn = EasyContrastiveLoss().to(device) if args.use_easy_contrastive else None
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        train_dataset = DerainDataset(root_dir=args.data_dir, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)#用 4 个子进程来加载数据
        test_dataset = DerainDataset(root_dir=args.data_dir, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        best_psnr = 0.0
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_one_epoch(train_loader, model, optimizer, pixel_loss_fn, perceptual_loss_fn, easy_contrastive_fn, args, device)

            current_psnr = evaluate(test_loader, model, device, lpips_fn)

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                checkpoint_data = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                model_name = f"{args.model}_perceptual.pth.tar" if args.use_perceptual_loss else f"{args.model}.pth.tar"
                save_checkpoint(checkpoint_data, filename=f"best_{model_name}")

            save_some_examples(model, test_loader, epoch, folder="evaluation_images", device=device)

    elif args.mode == "test":
        if not args.checkpoint: raise ValueError("Must provide a checkpoint for test mode.")
        load_checkpoint(torch.load(args.checkpoint, map_location=device), model, optimizer)
        test_dataset = DerainDataset(root_dir=args.data_dir, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        evaluate(test_loader, model, device, lpips_fn)


if __name__ == "__main__":
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    main()