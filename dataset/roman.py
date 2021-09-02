import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch

class Roman(Dataset):
  def __init__(self):
      import torchvision
      transform = torchvision.transforms.Compose(
          [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
      self.romandataset = torchvision.datasets.ImageFolder(root="./roman_rana/all", transform=transform)

  def __getitem__(self, idx):
    img, label = self.romandataset[idx]
    onehot_label = torch.eye(10)[label]
    return img, onehot_label
