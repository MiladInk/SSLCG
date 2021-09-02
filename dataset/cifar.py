import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch


# taken from https://github.com/PyTorchLightning/lightning-bolts/blob/c3b60de7dc30c5f7947256479d9be3a042b8c182/pl_bolts/transforms/dataset_normalizations.py#L20
def cifar10_normalization():
  normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
  )
  return normalize


tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])


class SSLCIFAR10(Dataset):
  def __init__(self, train):
    super(SSLCIFAR10, self).__init__()
    if train is True:
      cifar10_ssl_file = './cifar10_train_ssl.npz'
    else:
      cifar10_ssl_file = './cifar10_test_ssl.npz'

    self.dataset = datasets.CIFAR10(root='./', train=train, transform=tf, download=True)
    cifarssl = np.load(cifar10_ssl_file)
    self.labels = cifarssl['ssl_labels']
    self.rs = cifarssl['ssl_rs']

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    img, real_label = self.dataset[idx]
    ssl_label = self.labels[idx]
    ssl_label = torch.eye(10)[ssl_label]
    ssl_r = self.rs[idx]
    return img, ssl_label
