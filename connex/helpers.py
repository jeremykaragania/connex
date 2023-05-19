import math
import numpy as np

def visit_softmax_temperature(num_moves):
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

def select_action(action_space_size, num_moves, n, model):
  visit_counts = np.zeros(action_space_size)
  for i, j in n.children.items():
    visit_counts[i] = j.visit_count
  temperature = visit_softmax_temperature(num_moves)
  return softmax_sample(visit_counts, temperature)

def ucb_score(parent, child, stats, base=19652, init=1.25, discount=0.95):
  pb_c = math.log((parent.visit_count + base + 1) / base) + init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + discount * stats.normalize(child.value())
  else:
    value_score = 0
  return prior_score + value_score

def expand_node(n, to_play, actions, output):
  n.to_play = to_play
  n.state = output.state
  n.reward = output.reward
  policy = ({i: math.exp(output.policy[i]) for i in actions})
  policy_sum = sum(policy)
  for i, j in policy.items():
    n.children[i] = node(j / policy_sum)

def backpropagate(search_path, value, to_play, stats, discount=0.95):
  for i in reversed(search_path):
    i.value_sum += value if i.to_play == to_play else -value
    i.visit_count += 1
    stats.update(i.value())
    value = i.reward + discount * value

def add_exploration_noise(n, alpha=0.3, frac=0.25):
  actions = list(n.children.keys())
  noise = np.random.dirichlet([alpha] * len(actions))
  for i, j in zip(actions, noise):
    n.children[i].prior = n.children[i].prior * (1 - frac) + j * frac

def run_mcts(root, action_history, model, action_space_size, num_simulations=64):
  stats = min_max_stats()
  for i in range(num_simulations):
    history = list(action_history)
    node = root
    search_path = [node]
    while node.is_expanded():
      action, node = select_child(node, stats)
      history.append(action)
      search_path.append(node)
    parent = search_path[-2]
    to_play = len(history) % 2 == 0
    model_output = model.recurrent_inference(parent.state, history[-1])
    expand_node(node, to_play, [j for j  in range(action_space_size)], model_output)
    backpropagate(search_path, model_output.value, to_play, stats)

class replay_buffer():
  def __init__(self, window_size, batch_size):
    self.window_size = window_size
    self.batch_size = batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps, td_steps):
    games = [self.sample_game() for i in range(self.batch_size)]
    game_pos = [(i, self.sample_position(i)) for i in games]
    return [(i.make_image(j), i.action_history[j:j+num_unroll_steps], i.make_target(j, num_unroll_steps, td_steps, i.to_play())) for (i, j) in game_pos]

  def sample_game(self):
    return np.random.choice(self.buffer)

  def sample_position(self, game):
    return np.random.choice(len(game.environment_history))

def softmax_sample(distribution, temperature):
  if temperature == 0:
    temperature = 1
  distribution = np.array(distribution) ** (1 / temperature)
  sample_temp = distribution/distribution.sum()
  return np.argmax(np.random.multinomial(1, sample_temp))
