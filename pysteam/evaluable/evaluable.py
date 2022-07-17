import abc
import numpy as np
from collections import UserDict


class Node:
  """Class for a node in a block-automatic evaluation tree."""

  def __init__(self, value=None, *children):
    self.value = value
    self.children = list(children)


class Jacobians(UserDict):

  def add(self, key, value):
    if key in self.keys():
      self[key] += value
    else:
      self[key] = value


class Evaluable(abc.ABC):
  """Base class that defines the general 'evaluator' interface."""

  def evaluate(self, lhs: np.ndarray = None, jacs: Jacobians = None):
    """Interface for the general 'evaluation', optionally with Jacobians."""
    end_node = self.forward()
    if lhs is not None:
      self.backward(lhs, end_node, jacs)
    return end_node.value

  @property
  @abc.abstractmethod
  def active(self) -> bool:
    """Returns whether or not an evaluator contains unlocked state variables."""

  @property
  @abc.abstractmethod
  def related_var_keys(self) -> set:
    """Returns a set of state variables that are involved in the evaluation."""

  @abc.abstractmethod
  def forward(self) -> Node:
    """Forward pass with operation recorded"""

  @abc.abstractmethod
  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    """Backward pass to compute jacobian"""
