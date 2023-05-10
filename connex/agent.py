import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

model_output = collections.namedtuple("model_output", ["reward", "state", "policy", "value"])

class residual_block(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, padding=1)

  def forward(self, x):
    return F.relu(self.conv(x) + x)

class representation_function(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution = nn.Conv2d(2, 2, 3, padding=1)
    self.residual_blocks = nn.ModuleList([residual_block(2) for i in range(2)])

  def forward(self, observation):
    s = F.relu(self.convolution(torch.from_numpy(observation)))
    for i in self.residual_blocks:
      s = i(s)
    return s
