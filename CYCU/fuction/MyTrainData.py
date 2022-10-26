# -*- coding: utf-8 -*-
"""

@author: jia_2
"""
import torch

class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out