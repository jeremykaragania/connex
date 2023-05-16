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

class prediction_function(nn.Module):
  def __init__(self, action_space_size, environment_size):
    super().__init__()
    self.p_convolutions = nn.ModuleList([nn.Conv2d(2, 4, 1), nn.Conv2d(4, 1, 1)])
    self.p_linear = nn.Linear(environment_size, action_space_size)
    self.v_convolutions = nn.ModuleList([nn.Conv2d(2, 4, 1), nn.Conv2d(4, 1, 1)])
    self.v_linear = nn.Linear(environment_size, 1)

  def forward(self, state):
    p = F.relu(self.p_convolutions[0](state))
    p = self.p_convolutions[1](p)
    p = F.softmax(self.p_linear(p.flatten()), 0)
    v = self.v_convolutions[0](state)
    v = self.v_convolutions[1](v)
    v = torch.tanh(self.v_linear(v.flatten()))
    return p, v
