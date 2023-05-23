import environment
import helpers
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

model_output = collections.namedtuple("model_output", ["reward", "state", "policy", "value"])

class configuration():
  def __init__(self, training_steps, checkpoint_interval, num_unroll_steps, td_steps, learning_rate, weight_decay):
    self.training_steps = training_steps
    self.checkpoint_interval = checkpoint_interval
    self.num_unroll_steps = num_unroll_steps
    self.td_steps = td_steps
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

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

class dynamics_function(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.s_convolution = nn.Conv2d(4, 2, 3, padding=1)
    self.s_residual_blocks = nn.ModuleList([residual_block(2) for i in range(2)])
    self.r_convolution = nn.Conv2d(2, 1, 3, padding=1)
    self.r_linear = nn.Linear(config.environment_size(), 1)

  def forward(self, state, action):
    s = torch.cat([state, action])
    s = self.s_convolution(s)
    for i in self.s_residual_blocks:
      s = i(s)
    r = self.r_convolution(s)
    r = torch.tanh(self.r_linear(r.flatten()))
    return r, s

class prediction_function(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.p_convolutions = nn.ModuleList([nn.Conv2d(2, 4, 1), nn.Conv2d(4, 1, 1)])
    self.p_linear = nn.Linear(config.environment_size(), config.action_space_size)
    self.v_convolutions = nn.ModuleList([nn.Conv2d(2, 4, 1), nn.Conv2d(4, 1, 1)])
    self.v_linear = nn.Linear(config.environment_size(), 1)

  def forward(self, state):
    p = F.relu(self.p_convolutions[0](state))
    p = self.p_convolutions[1](p)
    p = F.softmax(self.p_linear(p.flatten()), 0)
    v = self.v_convolutions[0](state)
    v = self.v_convolutions[1](v)
    v = torch.tanh(self.v_linear(v.flatten()))
    return p, v

class model(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.representation = representation_function()
    self.dynamics = dynamics_function(config)
    self.prediction = prediction_function(config)

  def initial_inference(self, image):
    s = self.representation(image)
    p, v = self.prediction(s)
    return model_output(0, s, p, v.item())

  def recurrent_inference(self, state, action):
    action = torch.from_numpy(np.full(state.shape, action))
    r, s = self.dynamics(state, action)
    p, v = self.prediction(s)
    return model_output(r, s, p, v.item())

def play_game(config, model):
  game = environment.k_in_a_row(config)
  while not game.is_terminal():
    root = helpers.node(0)
    observation = game.make_image(-1)
    helpers.expand_node(root, game.to_play(), game.legal_actions(), model.initial_inference(observation))
    helpers.add_exploration_noise(root, 0.3)
    helpers.run_mcts(root, game.action_history, model, config.action_space_size)
    action = helpers.select_action(config.action_space_size, len(game.action_history), root, model)
    game.apply(action)
    game.store_search_statistics(root)
  return game
