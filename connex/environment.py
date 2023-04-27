import numpy as np

class k_in_a_row():
  def __init__(self, rows, columns, row_length):
    self.row_length = row_length
    self.environment = np.zeros((rows, columns), dtype=int)
    self.history = []

  def rows(self):
    return self.environment.shape[0]

  def columns(self):
    return self.environment.shape[1]

  def terminal(self):
    if len(self.history) == self.environment.size:
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
    self.history.append(action)

  def to_play(self):
    if len(self.history) % 2 == 0:
      return 1
    return -1
