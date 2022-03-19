from typing import Dict
import numpy as np

from ..state_key import StateKey
from ..evaluable import Evaluable, Node


class NegationEvaluator(Evaluable):
  """Evaluator for the negation of a vector."""

  def __init__(self, value: Evaluable) -> None:
    super().__init__()
    self._value: Evaluable = value

  @property
  def active(self) -> bool:
    return self._value.active

  def forward(self) -> Node:
    child = self._value.forward()
    value = -child.value
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._value.active:
      lhs = -lhs
      jacs = self._value.backward(lhs, node.children[0])
    return jacs

neg = NegationEvaluator

class AdditionEvaluator(Evaluable):
  """Evaluator for the addition of two vectors."""

  def __init__(self, lhs: Evaluable, rhs: Evaluable) -> None:
    super().__init__()
    self._lhs: Evaluable = lhs
    self._rhs: Evaluable = rhs

  @property
  def active(self) -> bool:
    return self._lhs.active or self._rhs.active

  def forward(self) -> Node:
    lhs = self._lhs.forward()
    rhs = self._rhs.forward()
    value = lhs.value + rhs.value
    return Node(value, lhs, rhs)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()

    if self._lhs.active:
      jacs = self._lhs.backward(lhs, node.children[0])

    if self._rhs.active:
      jacs2 = self._rhs.backward(lhs, node.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs

add = AdditionEvaluator

class ScalarMultEvaluator(Evaluable):

  def __init__(self, value: Evaluable, scalar: float) -> None:
    super().__init__()
    self._value: Evaluable = value
    self._scalar: float = scalar

  @property
  def active(self) -> bool:
    return self._value.active

  def forward(self) -> Node:
    child = self._value.forward()
    value = self._scalar * child.value
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._value.active:
      lhs = self._scalar * lhs
      jacs = self._value.backward(lhs, node.children[0])
    return jacs

smult = ScalarMultEvaluator