import os, glob, shutil, random
import torch,cv2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.autograd.profiler as profiler
import torch.nn.functional as F

from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
from torch import nn
from torch.nn import Module
from torch.optim import RMSprop


class MriSegmentation(Dataset):
    """MRI dataset."""

    def __init__(self, root_dir, transform=None):
        self.images = []
        self.masks = []
        self.root_dir = root_dir

        all_images = []
        all_mask = []
        for i in os.listdir(root_dir):
            if i.startswith("TCGA_"):
                folder_path = os.path.join(root_dir, i)
                for file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, file)
                    if img_path.endswith("_mask.tif"):
                        img = cv2.imread(img_path, 0)
                        all_mask.append((img_path, img))
                    else:
                        img = cv2.imread(img_path)
                        img = img[:, :, ::-1]
                        all_images.append((img_path, img))

        for i, a in all_images:
            for j, b in all_mask:
                if i.split(".tif")[0] == j.split("_mask.tif")[0]:
                    self.images.append(a)
                    self.masks.append(b // 255)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Get a dict of the pair
        """

        image = self.images[index]
        label = self.masks[index]
        label = cv2.resize(label, (68, 68))

        image = torch.from_numpy(image.copy()).view(3, 256, 256)
        label = torch.from_numpy(label.copy()).view(1, 68, 68)

        return image.type(torch.FloatTensor), label.type(torch.FloatTensor)


if __name__ == "__main__":
    mri_dataset = MriSegmentation(root_dir="C://Users//Pragya//varied_projects//kaggle_3m", transform=transforms.Compose([transforms.Resize((256,256)),
                                                                                      transforms.RandomHorizontalFlip(),
                                                                                      transforms.RandomVerticalFlip(),
                                                                                      transforms.RandomRotation(90),
                                                                                      transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 0.8)),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize(mean=[0.5], std=[0.5])
                                                                                     ]))
    print(mri_dataset)
    # dataloader = DataLoader(mri_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=None)

    n_val = int(len(mri_dataset) * 0.2)
    n_train = len(mri_dataset) - n_val
    train, val = random_split(mri_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=4, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    for data, label in train_loader:
        print(type(data))
        print(data.shape)
        print(type(label))
        print(label.shape)
        break