import abc
import operator
from typing import Optional, Dict
import numpy as np

from ..state import StateKey


class Evaluator(abc.ABC):
  """Base class that defines the general 'evaluator' interface."""

  @abc.abstractmethod
  def is_active(self) -> bool:
    """Returns whether or not an evaluator contains unlocked state variables."""

  @abc.abstractmethod
  def evaluate(self, lhs: Optional[np.ndarray] = None):
    """Interface for the general 'evaluation', optionally with Jacobians when lhs if not None."""


class EvalTreeNode:
  """Class for a node in a block-automatic evaluation tree."""

  def __init__(self, value=None, *children):
    self.value = value
    self.children = list(children)


class AutoGradEvaluator(Evaluator):
  """Base class that defines the general 'auto-grad evaluator' interface."""

  def evaluate(self, lhs: np.ndarray = None):
    """Interface for the general 'evaluation', optionally with Jacobians."""
    # forward pass
    tree = self.get_eval_tree()

    if lhs is None:
      return tree.value

    # backward pass
    jacs = self.compute_jacs(lhs, tree)

    return tree.value, jacs

  @abc.abstractmethod
  def get_eval_tree(self) -> EvalTreeNode:
    """Interface for an evaluation method that returns the tree of evaluations (forward pass)."""

  @abc.abstractmethod
  def compute_jacs(self, lhs, tree) -> Dict[StateKey, np.ndarray]:
    """Interface for computing the Jacobian (backward pass)."""

  @staticmethod
  def merge_jacs(a, b, op=operator.add):
    """Utility function to merge Jabobians w.r.t. the same variables"""
    # Start with symmetric difference; keys either in a or b, but not both
    merged = {k: a.get(k, b.get(k)) for k in a.keys() ^ b.keys()}
    # Update with `op()` applied to the intersection
    merged.update({k: op(a[k], b[k]) for k in a.keys() & b.keys()})
    return merged
