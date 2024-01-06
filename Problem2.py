

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Default transforms
def transform_train(crop_size=224):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def transform_test():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


# Data loader for the image segmentation data
class ImageSegmentation(Dataset):
    def __init__(self, root, split, transform=None, device="cpu"):
        self.root = root
        self.split = split
        self.transform = transform
        self.device = device

        self.dir = os.path.join(root, split)
        self.camera_files = sorted(os.listdir(os.path.join(self.dir, "Camera")))
        if self.split != "Test":
            self.seg_files = sorted(os.listdir(os.path.join(self.dir, "Labels")))

    def __len__(self):
        return len(self.camera_files)

    # Some good ideas here would be to crop a smaller section of the image
    # And add random flipping
    # Make sure the same augmentation is applied to image and label
    def image_augmentation(self, img_mat, label_mat):
        # Convert NumPy arrays to PIL Image
        img = Image.fromarray(img_mat)
        label = Image.fromarray(label_mat)

        # Random horizontal flipping using PyTorch transform
        if np.random.rand() > 0.5:
            transform_flip = transforms.RandomHorizontalFlip(p=1)
            img = transform_flip(img)
            label = transform_flip(label)

        # Random cropping using PyTorch transform
        transform_crop = transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
        img = transforms.functional.crop(img, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)

        # Convert PIL Images back to NumPy arrays
        img_mat = np.array(img)
        label_mat = np.array(label)

        return img_mat, label_mat

    # Return indexed item in dataset
    def __getitem__(self, index):
        file_name = os.path.join(self.dir, "Camera", self.camera_files[index])
        img = Image.open(file_name)
        img_mat = np.copy(np.asarray(img)[:, :, :3])
        if self.split != "Test":
            labeled_img = Image.open(os.path.join(self.dir, "Labels", self.seg_files[index]))
            label_mat = np.copy(np.asarray(labeled_img)[:, :, :3])
        else:
            label_mat = np.zeros_like(img_mat)
        if self.split == "Train":
            img_mat, label_mat = self.image_augmentation(img_mat, label_mat)
        return self.transform(img_mat), torch.tensor(label_mat, device=self.device)

    # Combine data within the batch
    def collate_fn(self, data):
        B = len(data)
        img_batch = torch.stack([data[i][0] for i in range(B)]).to(self.device)
        label_batch = torch.stack([data[i][1] for i in range(B)]).to(self.device)
        return img_batch, label_batch


# Basic convolution block with a 2D convolution, ReLU, and BatchNorm layer
# Conv with kernel size 2, stride 2, and padding 0 decreases the size of the image by half
# Conv with kernel size 3, stride 1, padding 1 keeps the image size constant
class ConvBlockStudent(nn.Module):
    def __init__(self, c_in, c_out, ds=False):
        super().__init__()
        if ds:
          self.net = nn.Sequential(
              nn.Conv2d(c_in, c_out, 2, stride=2, padding=0),
              nn.ReLU(),
              nn.BatchNorm2d(c_out),
          )
        else:
          self.net = nn.Sequential(
              nn.Conv2d(c_in, c_out, 3, stride=1, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(c_out),
          )

    def forward(self, x):
        return self.net(x)


# This is a basic U Net class. The decoder downsamples the image resolution at each level
# The encoder fuses information from the same resolution from the encoder at each level
# With a convolution operation. 

# In the encoder, we perform upsampling to ensure the same resolution with simple
# bilinear interpolation. An alternative to this is transposed convolution: 
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
class UNetStudent(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pre = ConvBlockStudent(3, 16)

        self.down1 = ConvBlockStudent(16, 32, ds=True)

        self.down2 = ConvBlockStudent(32, 64, ds=True)

        self.up1 = ConvBlockStudent(64+32, 32)

        self.up0 = ConvBlockStudent(32+16, 32)

        self.out = nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x0 = self.pre(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        # Going up 1st layer
        B, __, H, W = x1.shape
        x2 = F.interpolate(x2, (H, W))
        x2 = torch.cat([x1, x2], dim=1)
        x1 = self.up1(x2)

        # Going up 0th layer
        B, __, H, W = x0.shape
        x1 = F.interpolate(x1, (H, W))
        x1 = torch.cat([x0, x1], dim=1)
        x = self.up0(x1)
        return self.out(x)


# Compute the per-class iou and miou
def IoU(targets, predictions, num_classes, ignore_index=0):
  intersections = torch.zeros(num_classes, device=targets.device)
  unions = torch.zeros_like(intersections)
  counts = torch.zeros_like(intersections)
  # TODO: Discard ignored points
  valid_mask = targets != ignore_index #None
  targets = targets[valid_mask]
  predictions = predictions[valid_mask]
  # Loop over classes and update the counts, unions, and intersections
  for c in range(num_classes):
    t_c = (targets == c)
    p_c = (predictions == c)
    intersections[c] = torch.sum(t_c & p_c).float()
    unions[c] = torch.sum(t_c | p_c).float()
    counts[c] = torch.sum(t_c).float()
    # TODO: Fill in computation
    # Add small value to avoid division by 0
    # Make sure to keep the small smoothing constant to match the autograder
    unions[c] = unions[c] + 0.00001
  # Per-class IoU
  # Make sure to set iou for classes with no points to 1
  iou = intersections / unions #None
  iou[counts == 0] = 1.0
  # Calculate mean, ignoring ignore index
  total_iou = torch.sum(iou[1:])
  total_classes = num_classes - 1 
  miou = total_iou / total_classes#None
  return iou, miou

    
