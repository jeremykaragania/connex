import numpy as np

class configuration():
  def __init__(self, rows, columns, row_length):
    self.rows = rows
    self.columns = columns
    self.row_length = row_length
    self.action_space_size = columns
    self.environment_size = lambda: self.rows * self.columns

class k_in_a_row():
  def __init__(self, config):
    self.row_length = config.row_length
    self.environment = np.zeros((config.rows, config.columns), dtype=int)
    self.rewards = []
    self.action_history = []
    self.environment_history = [np.copy(self.environment)]
    self.child_visits = []
    self.root_values = []

  def rows(self):
    return self.environment.shape[0]

  def columns(self):
    return self.environment.shape[1]

  def is_terminal(self):
    if len(self.action_history) == self.environment.size:
      return True
    lines = (
      lambda x, y, d: self.environment[x+d][y],
      lambda x, y, d: self.environment[x][y+d],
      lambda x, y, d: self.environment[x+d][y+d],
      lambda x, y, d: self.environment[x+d][y-d] if y - d > 0 else 0
    )
    for i in range(self.rows()):
      for j in range(self.columns()):
        player = self.environment[i][j]
        if player == 0:
          continue
        for k in lines:
          for l in range(self.row_length):
            try:
              if k(i, j, l) != player:
                break
            except IndexError:
              break
            if l == self.row_length - 1:
              return True
    return False

  def legal_actions(self):
    return np.array([i for i in range(self.columns()) if self.environment[0][i] == 0])

  def apply(self, action):
    for i in reversed(range(self.rows())):
      if self.environment[i][action] == 0:
        self.environment[i][action] = self.to_play()
        break
    reward = 0
    if self.is_terminal() and len(self.action_history) != self.environment.size:
      reward = self.to_play()
    elif self.to_play() and action not in self.legal_actions():
      reward = -1
    self.rewards.append(reward)
    self.action_history.append(action)
    self.environment_history.append(np.copy(self.environment))

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = [i for i in range(self.columns())]
    self.child_visits.append([root.children[i].visit_count / sum_visits if i in root.children else 0 for i in action_space])
    self.root_values.append(root.value())

  def make_image(self, state_index):
    players = (np.where(self.environment_history[state_index] == 1, 1, 0), np.where(self.environment_history[state_index] == -1, 1, 0))
    return np.array([players[0], players[1]], dtype=np.float32)

  def make_target(self, state_index, num_unroll_steps, td_steps, discount=0.95):
    targets = []
    for i in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = i + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * discount ** td_steps
      else:
        value = 0
      for j, k in enumerate(self.rewards[i:bootstrap_index]):
        value += k * discount ** j
      if i > 0 and i <= len(self.rewards):
        last_reward = self.rewards[i-1]
      else:
        last_reward = 0
      if i < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[i]))
      else:
        targets.append((0, last_reward, []))
    return targets

  def to_play(self):
    if len(self.action_history) % 2 == 0:
      return 1
    return -1
