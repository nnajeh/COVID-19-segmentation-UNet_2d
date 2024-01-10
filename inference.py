import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from model import UNet


model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        #
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)


y1 = y[0, 0].cpu().detach().numpy()
y2 = y_pred[0, 0].cpu().detach().numpy()

plt.subplot(121)
plt.imshow(y1, cmap='gray')

plt.subplot(122)
plt.imshow(y2, cmap='gray')
plt.show()

y_color = np.zeros((*y2.shape, 3))
y_color[..., 0] = y1
y_color[..., 1] = y2
y_color[..., 2] = y2


plt.subplot(121)
plt.imshow(X[0, 0].cpu().detach().numpy())

plt.subplot(122)
plt.imshow(y_color)



import torch
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'test_loader' is your DataLoader
batch = next(iter(test_loader))

with torch.no_grad():
    model.eval()
    logits = model(batch[0].to(device))
pr_masks = (logits.squeeze(1) > 0.5).float()

for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(15, 5))  # Increase the width for better visualization

    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Use permute for CHW to HWC conversion
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    # Convert RGB to grayscale for ground truth mask
    grayscale_gt_mask = np.mean(gt_mask.squeeze().cpu().numpy(), axis=0)
    plt.imshow(grayscale_gt_mask, cmap='gray')
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # Convert RGB to grayscale for prediction mask
    if pr_mask.shape[0] == 3:
        grayscale_pr_mask = np.mean(pr_mask.cpu().numpy(), axis=0)
        plt.imshow(grayscale_pr_mask, cmap='gray')
    else:
        plt.imshow(pr_mask.cpu().numpy(), cmap='gray')
    
    plt.title("Prediction")
    plt.axis("off")
    plt.show()





# Assuming 'test_loader' is your DataLoader
batch = next(iter(test_loader))

with torch.no_grad():
    model.eval()
    logits = model(batch[0].to(device))  # Assuming image is at index 0
pr_masks = (logits.squeeze(1) > 0.5).float()

# Iterate through the batches
for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    grayscale_gt_mask = np.mean(gt_mask.squeeze().cpu().numpy(), axis=0)
    plt.imshow(grayscale_gt_mask, cmap='gray')
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if pr_mask.shape[0] == 3:
        grayscale_pr_mask = np.mean(pr_mask.cpu().numpy(), axis=0)
        plt.imshow(grayscale_pr_mask, cmap='gray')
    else:
        plt.imshow(pr_mask.cpu().numpy(), cmap='gray')
    
    plt.title("Prediction")
    plt.axis("off")
    plt.show()
