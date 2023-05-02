import math
import numpy as np

class minimax_stats():
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
