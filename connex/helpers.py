import math
import numpy as np

def visit_softmax_temperature(num_moves, training_steps):
  return 1 if num_moves > 30 else 0

class min_max_stats():
  def __init__(self):
    self.maximum = -float("inf")
    self.minimum= float("inf")

  def update(self, value):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value):
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class node():
  def __init__(self, prior): 
    self.visit_count = 0
    self.to_play = 1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.state = None
    self.reward = 0

  def is_expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

def select_child(n, stats):
  action, child = max((ucb_score(n, j, stats), i, j) for i, j in n.children.items())[1:]
  return action, child

def ucb_score(parent, child, stats):
  base = 19652
  init = 1.25
  discount = 0.95
  pb_c = math.log((parent.visit_count + base + 1) / base) + init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + discount * stats.normalize(child.value())
  else:
    value_score = 0
  return prior_score + value_score

def expand_node(n, to_play, actions, output):
  discount = 0.95
  n.to_play = to_play
  n.state = output.state
  n.reward = output.reward
  policy = ({i: math.exp(output.policy[i]) for i in actions})
  policy_sum = sum(policy)
  for i, j in policy.items():
    n.children[i] = node(j / policy_sum)

def backpropagate(search_path, value, to_play, stats):
  discount = 0.95
  for i in reversed(search_path):
    i.value_sum += value if i.to_play == to_play else -value
    i.visit_count += 1
    stats.update(i.value())
    value = i.reward + discount * value

def add_exploration_noise(n, alpha):
  actions = list(n.children.keys())
  frac = 0.25
  noise = np.random.dirichlet([alpha] * len(actions))
  for i, j in zip(actions, noise):
    n.children[i].prior = n.children[i].prior * (1 - frac) + j * frac
