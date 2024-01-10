import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm
from model import UNet


# Define the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## Load dataset
import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_msk=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_msk = transform_msk
        # Define the transformations to be applied to the images and masks
        
        
        self.transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[1,1,1])

])

        self.transform_msk = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[1,1,1])

])

        self.image_paths = sorted(glob(os.path.join(self.image_dir, '*.png')))
        self.mask_paths = sorted(glob(os.path.join(self.mask_dir, '*.png')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask file not found at index {idx}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform_img(image)

        mask = (Image.open(mask_path))
        mask = self.transform_msk(mask)



        return image, mask

# Define the paths to the training and testing data
train_image_dir = "./new-dataset-covid19/images_train/images_train/"
train_mask_dir = "./new-dataset-covid19/annotations_train/annotations_train/"
val_image_dir = "./new-dataset-covid19/images_val/images_val/"
val_mask_dir = "./new-dataset-covid19/annotations_val/annotations_val/"
test_image_dir = "./new-dataset-covid19/images_test/images_test/"
test_mask_dir = "./new-dataset-covid19/annotations_test/annotations_test/"

# Define the training and testing datasets
train_dataset = CustomDataset(train_image_dir, train_mask_dir)
test_dataset = CustomDataset(test_image_dir, test_mask_dir)
val_dataset = CustomDataset(val_image_dir, val_mask_dir)

# Define the training and testing data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)



## Define diceloss
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


import torch.optim as optim
epochs=100
# Setup loss function and optimizer
criterion = nn.BCEWithLogitsLoss()# DiceLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Define model
model = UNet().to(device)

# Setup loss function and optimizer
loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Set the seed and start the timer
torch.manual_seed(42)

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 300
train_total_losses = []
val_total_losses = []

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    train_losses, test_losses = [], []
    print(f"Epoch: {epoch+1} of {epochs}")

    ### Training
    train_loss_1, train_loss_2, train_loss = 0, 0, 0

    model.train()

    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        train_loss += loss.item()  # accumulate the loss per epoch
        train_loss_1 += loss_1.item()
        train_loss_2 += loss_2.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        train_losses.append(loss.item())

    # Divide total train loss by the length of the train dataloader (average loss per batch per epoch)
    train_loss /= len(train_loader)
    train_loss_1 /= len(train_loader)
    train_loss_2 /= len(train_loader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss_1, test_loss_2, test_loss = 0, 0, 0
    model.eval()

    for X, y in val_loader:
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (accumulatively)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        test_loss += loss.item()
        test_loss_1 += loss_1.item()
        test_loss_2 += loss_2.item()

        test_losses.append(loss.item())

    # Calculations on test metrics need to happen outside the loop
    # Divide total test loss by the length of the test dataloader (per batch)
    test_loss /= len(val_loader)
    test_loss_1 /= len(val_loader)
    test_loss_2 /= len(val_loader)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.5f}, Dice: {train_loss_1:.5f}, BCE: {train_loss_2:.5f} | Test loss: {test_loss:.5f}, Dice: {test_loss_1:.5f}, BCE: {test_loss_2:.5f}\n")

    train_loss = np.average(train_losses)
    train_total_losses.append(train_loss)

    val_loss = np.average(test_losses)
    val_total_losses.append(test_loss)

    if epoch % 5 == 0:
        plt.subplot(231)
        plt.imshow(X[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(y[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(y_pred[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(X[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(y[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(y_pred[0, 0].cpu().detach().numpy())
        plt.axis('off')
        plt.show()

    if epoch % 5 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"./model-{epoch}.pth")
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_total_losses, label='train_loss')
        plt.plot(val_total_losses, label='val_loss')
        plt.title("Training & Validation Losses")
        plt.ylabel(" Losses")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
