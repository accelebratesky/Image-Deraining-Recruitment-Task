import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch
class DerainDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=128):#默认为训练
        self.root_dir = root_dir
        self.is_train = is_train

        data_folder = 'train' if is_train else 'test'
        # 拼接路径：rainy图在 input，干净图在 target
        self.rainy_dir = os.path.join(self.root_dir, data_folder, 'input')
        self.clean_dir = os.path.join(self.root_dir, data_folder, 'target')
        #sorted()排序，把os.listdir得到的乱序文件名，按字母或数字顺序排好,保证一一配对
        self.rainy_images = sorted(os.listdir(self.rainy_dir))

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()#numpy转tensor+归一化
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.rainy_images[idx])

        rainy_image = Image.open(rainy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        rainy_tensor = self.transform(rainy_image)
        clean_tensor = self.transform(clean_image)
        if self.is_train and random.random() < 0.5:
            rainy_tensor = torch.flip(rainy_tensor, dims=[2])
            clean_tensor = torch.flip(clean_tensor, dims=[2])#用相同的随机条件一起翻转，保证雨图和干净图的翻转状态完全一致
        return rainy_tensor, clean_tensor, self.rainy_images[idx]