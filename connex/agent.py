import connex.environment as environment
import connex.helpers as helpers
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

model_output = collections.namedtuple("model_output", ["reward", "state", "policy", "value"])

class configuration:
  def __init__(self, training_steps, checkpoint_interval, window_size, batch_size, num_unroll_steps, td_steps, learning_rate, weight_decay):
    self.training_steps = training_steps
    self.checkpoint_interval = checkpoint_interval
    self.window_size = window_size
    self.batch_size = batch_size
    self.num_unroll_steps = num_unroll_steps
    self.td_steps = td_steps
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

class convolution(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    ret = self.conv(x)
    return self.bn(ret)

class residual_block(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = convolution(channels, channels, 3, padding=1)

  def forward(self, x):
    return F.relu(self.conv(x) + x)

class representation_function(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution = convolution(2, 2, 3, padding=1)
    self.residual_blocks = nn.ModuleList([residual_block(2) for i in range(2)])

  def forward(self, observation):
    s = F.relu(self.convolution(torch.from_numpy(observation)))
    for i in self.residual_blocks:
      s = i(s)
    return s

class dynamics_function(nn.Module):
  def __init__(self, game_config):
    super().__init__()
    self.s_convolution = convolution(4, 2, 3, padding=1)
    self.s_residual_blocks = nn.ModuleList([residual_block(2) for i in range(2)])
    self.r_convolution = convolution(2, 1, 3, padding=1)
    self.r_linear = nn.Linear(game_config.environment_size(), 1)

  def forward(self, state, action):
    s = torch.cat([state, action], dim=1)
    s = self.s_convolution(s)
    for i in self.s_residual_blocks:
      s = i(s)
    r = self.r_convolution(s)
    r = torch.tanh(self.r_linear(r.flatten()))
    return r, s

class prediction_function(nn.Module):
  def __init__(self, game_config):
    super().__init__()
    self.p_convolutions = nn.ModuleList([convolution(2, 4, 1), convolution(4, 1, 1)])
    self.p_linear = nn.Linear(game_config.environment_size(), game_config.action_space_size)
    self.v_convolutions = nn.ModuleList([convolution(2, 4, 1), convolution(4, 1, 1)])
    self.v_linear = nn.Linear(game_config.environment_size(), 1)

  def forward(self, state):
    p = F.relu(self.p_convolutions[0](state))
    p = self.p_convolutions[1](p)
    p = F.softmax(self.p_linear(p.flatten()), 0)
    v = self.v_convolutions[0](state)
    v = self.v_convolutions[1](v)
    v = torch.tanh(self.v_linear(v.flatten()))
    return p, v

class model(nn.Module):
  def __init__(self, game_config):
    super().__init__()
    self.representation = representation_function()
    self.dynamics = dynamics_function(game_config)
    self.prediction = prediction_function(game_config)

  def initial_inference(self, image):
    s = self.representation(image)
    p, v = self.prediction(s)
    return model_output(0, s, p, v.item())

  def recurrent_inference(self, state, action):
    action = torch.from_numpy(np.full(state.shape, action))
    r, s = self.dynamics(state, action)
    p, v = self.prediction(s)
    return model_output(r, s, p, v.item())

def play_game(game_config, model):
  game = environment.k_in_a_row(game_config)
  while not game.is_terminal():
    root = helpers.node(0)
    observation = game.make_image(-1)
    helpers.expand_node(root, game.to_play(), game.legal_actions(), model.initial_inference(observation))
    helpers.add_exploration_noise(root, 0.3)
    helpers.run_mcts(root, game.action_history, model, game_config.action_space_size)
    action = helpers.select_action(game_config.action_space_size, len(game.action_history), root, model)
    game.apply(action)
    game.store_search_statistics(root)
  return game

def run_selfplay(game_config, storage, replay_buffer):
  while True:
    m = storage[-1]
    game = play_game(game_config, m)
    replay_buffer.save_game(game)

def train_model(game_config, model_config, storage, replay_buffer, verbose=False):
  m = model(game_config)
  while True:
    optimizer = optim.SGD(m.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
    for i in range(model_config.training_steps):
      if replay_buffer.buffer:
        if i % model_config.checkpoint_interval == 0:
          storage.append(m)
        batch = replay_buffer.sample_batch(model_config.num_unroll_steps, model_config.td_steps)
        update_parameters(optimizer, m, batch, verbose)
      storage.append(m)

def update_parameters(optimizer, m, batch, verbose=False):
  m.train()
  p_loss = 0
  v_loss = 0
  r_loss = 0
  for image, actions, targets in batch:
    reward, state, policy, value = m.initial_inference(image)
    predictions = [(1, value, reward, policy)]
    for action in actions:
      reward, state, policy, value = m.recurrent_inference(state, action)
      predictions.append((1 / len(actions), value, reward, policy))
    for prediction, target in zip(predictions, targets):
      value, reward, policy = prediction[1:]
      target_value, target_reward, target_policy = target
      if target_policy:
        p_loss += torch.sum(-torch.tensor(target_policy) * torch.log(policy))
        v_loss += torch.sum((torch.tensor([target_value]) - value) ** 2)
        r_loss += torch.sum((torch.tensor([target_reward]) - value) ** 2)
  optimizer.zero_grad()
  loss = p_loss + v_loss + r_loss
  loss.backward()
  optimizer.step()
  if verbose:
    print(f"Loss:")
    print(f"{'  Policy:':<16}{p_loss}")
    print(f"{'  Value:':<16}{v_loss}")
    print(f"{'  Reward:':<16}{r_loss}")
