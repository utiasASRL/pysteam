from typing import Dict
import numpy as np

from pylgmath import se3op

from ..state_key import StateKey
from ..evaluatable import Evaluatable, Node


class LogMapEvaluator(Evaluatable):
  """Evaluator for the logarithmic map of a transformation matrix."""

  def __init__(self, transform: Evaluatable) -> None:
    super().__init__()
    self._transform: Evaluatable = transform

  @property
  def active(self) -> bool:
    return self._transform.active

  def forward(self) -> Node:
    child = self._transform.forward()
    value = child.value.vec()
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._transform.active:
      lhs = lhs @ se3op.vec2jacinv(node.value)
      jacs = self._transform.backward(lhs, node.children[0])
    return jacs


class InverseEvaluator(Evaluatable):
  """Evaluator for the inverse of a transformation matrix"""

  def __init__(self, transform: Evaluatable) -> None:
    super().__init__()
    self._transform: Evaluatable = transform

  @property
  def active(self) -> bool:
    return self._transform.active

  def forward(self) -> Node:
    child = self._transform.forward()
    value = child.value.inverse()
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._transform.active:
      lhs = -lhs @ node.value.adjoint()
      jacs = self._transform.backward(lhs, node.children[0])
    return jacs


class ComposeEvaluator(Evaluatable):
  """Evaluator for the composition of transformation matrices."""

  def __init__(self, transform1: Evaluatable, transform2: Evaluatable):
    super().__init__()
    self._transform1: Evaluatable = transform1
    self._transform2: Evaluatable = transform2

  @property
  def active(self) -> bool:
    return self._transform1.active or self._transform2.active

  def forward(self) -> Node:
    child1 = self._transform1.forward()
    child2 = self._transform2.forward()

    value = child1.value @ child2.value
    return Node(value, child1, child2)

  def backward(self, lhs, tree):
    jacs = dict()

    if self._transform1.active:
      jacs = self._transform1.backward(lhs, tree.children[0])

    if self._transform2.active:
      lhs = lhs @ tree.children[0].value.adjoint()
      jacs2 = self._transform2.backward(lhs, tree.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


class ComposeInverseEvaluatable(Evaluatable):
  """Evaluator for the composition of two transformation matrices (with one inverted)."""

  def __init__(self, transform1, transform2):
    super().__init__()
    self._transform1 = transform1
    self._transform2 = transform2

  @property
  def active(self) -> bool:
    return self._transform1.active or self._transform2.active

  def forward(self):
    child1 = self._transform1.forward()
    child2 = self._transform2.forward()

    value = child1.value @ child2.value.inverse()
    return Node(value, child1, child2)

  def backward(self, lhs, tree):
    jacs = dict()

    if self._transform1.active:
      jacs = self._transform1.backward(lhs, tree.children[0])

    if self._transform2.active:
      tf_ba = tree.children[0].value @ tree.children[1].value.inverse()
      lhs = -lhs @ tf_ba.adjoint()
      jacs2 = self._transform2.backward(lhs, tree.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs