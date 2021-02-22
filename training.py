import torch.nn as nn

from torch import optim
from torchvision import transforms, utils
from tqdm import tqdm

from dataloader import MriSegmentation
from model import Net
import torch,cv2

from torchvision import transforms, utils
from torch import nn
from torch.utils.data import  Dataset, DataLoader, random_split

if __name__ == '__main__':
    mri_dataset = MriSegmentation(root_dir="C://Users//Pragya//varied_projects//kaggle_3m",
                                  transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.RandomRotation(90),
                                                                transforms.RandomAffine(degrees=15,
                                                                                        translate=(0.1, 0.1),
                                                                                        scale=(0.8, 0.8)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.5], std=[0.5])]))

    n_val = int(len(mri_dataset) * 0.1)
    n_train = len(mri_dataset) - n_val
    train, val = random_split(mri_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=4, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # print(train_loader)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net(3, 1, bilinear=True)
    # net.to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-8)

    net.train()
    epoch_loss = 0
    for epoch in range(1):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            running_loss = 0
            # get the inputs; data is a list of [inputs, labels]
            images, masks = data
            # print("images : ", images.shape)
            # print("masks : ", masks.shape)
            # images = images.to(device=device)
            # masks = masks.to(device=device)
            output = net(images)
            print("output = ", output.shape)
            loss = criterion(output, masks)
            print("loss = ", loss)
            epoch_loss += loss.item()
            running_loss += loss.item()
            print("running_loss = ", running_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.sigmoid(output)
        print("epoch_loss = ", epoch_loss)

    PATH = './mrisegmentation_unet.pth'
    torch.save(net.state_dict(), PATH)
