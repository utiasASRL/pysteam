import abc
import operator
from typing import Dict
import numpy as np

from .state_key import StateKey


class Node:
  """Class for a node in a block-automatic evaluation tree."""

  def __init__(self, value=None, *children):
    self.value = value
    self.children = list(children)


class Evaluatable(abc.ABC):
  """Base class that defines the general 'evaluator' interface."""

  @property
  @abc.abstractmethod
  def active(self) -> bool:
    """Returns whether or not an evaluator contains unlocked state variables."""

  def evaluate(self, lhs: np.ndarray = None):
    """Interface for the general 'evaluation', optionally with Jacobians."""
    # forward pass
    end_node = self.forward()
    if lhs is None:
      return end_node.value
    else:
      return end_node.value, self.backward(lhs, end_node)

  @abc.abstractmethod
  def forward(self) -> Node:
    """Forward pass with operation recorded"""

  @abc.abstractmethod
  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    """Backward pass to compute jacobian"""

  @staticmethod
  def merge_jacs(a, b, op=operator.add):
    """Utility function to merge Jabobians w.r.t. the same variables"""
    # Start with symmetric difference; keys either in a or b, but not both
    merged = {k: a.get(k, b.get(k)) for k in a.keys() ^ b.keys()}
    # Update with `op()` applied to the intersection
    merged.update({k: op(a[k], b[k]) for k in a.keys() & b.keys()})
    return merged
