from typing import Dict
import numpy as np

from pylgmath import se3op, Transformation

from ..state_key import StateKey
from ..evaluable import Evaluable, Node


class ExpMapEvaluator(Evaluable):
  """Evaluator of ExpMap."""

  def __init__(self, value: Evaluable) -> None:
    super().__init__()
    self._value = value

  @property
  def active(self) -> bool:
    return self._value.active

  def forward(self) -> Node:
    child = self._value.forward()
    value = Transformation(xi_ab=child.value)
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._value.active:
      lhs = lhs @ se3op.vec2jac(node.value.vec())
      jacs = self._value.backward(lhs, node.children[0])
    return jacs


vec2tran = ExpMapEvaluator


class LogMapEvaluator(Evaluable):
  """Evaluator for the logarithmic map of a transformation matrix."""

  def __init__(self, transform: Evaluable) -> None:
    super().__init__()
    self._transform: Evaluable = transform

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


tran2vec = LogMapEvaluator


class InverseEvaluator(Evaluable):
  """Evaluator for the inverse of a transformation matrix"""

  def __init__(self, transform: Evaluable) -> None:
    super().__init__()
    self._transform: Evaluable = transform

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


inv = InverseEvaluator


class ComposeEvaluator(Evaluable):
  """Evaluator for the composition of transformation matrices."""

  def __init__(self, transform1: Evaluable, transform2: Evaluable):
    super().__init__()
    self._transform1: Evaluable = transform1
    self._transform2: Evaluable = transform2

  @property
  def active(self) -> bool:
    return self._transform1.active or self._transform2.active

  def forward(self) -> Node:
    child1 = self._transform1.forward()
    child2 = self._transform2.forward()

    value = child1.value @ child2.value
    return Node(value, child1, child2)

  def backward(self, lhs, node):
    jacs = dict()

    if self._transform1.active:
      jacs = self._transform1.backward(lhs, node.children[0])

    if self._transform2.active:
      lhs = lhs @ node.children[0].value.adjoint()
      jacs2 = self._transform2.backward(lhs, node.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


compose = ComposeEvaluator


class ComposeInverseEvaluator(Evaluable):
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

  def backward(self, lhs, node):
    jacs = dict()

    if self._transform1.active:
      jacs = self._transform1.backward(lhs, node.children[0])

    if self._transform2.active:
      tf_ba = node.children[0].value @ node.children[1].value.inverse()
      lhs = -lhs @ tf_ba.adjoint()
      jacs2 = self._transform2.backward(lhs, node.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


compose_rinv = ComposeInverseEvaluator


class JacobianEvaluator(Evaluable):
  """Evaluator of Jacobian."""

  def __init__(self, value: Evaluable) -> None:
    super().__init__()
    self._value = value

  @property
  def active(self) -> bool:
    return self._value.active

  def forward(self) -> Node:
    child = self._value.forward()
    value = se3op.vec2jac(child.value)
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    print("WARNING: JacobianEvaluator.backward computation may be incorrect!")
    jacs = dict()
    if self._value.active:
      lhs = 2.0 * lhs  # TODO: check this
      jacs = self._value.backward(lhs, node.children[0])
    return jacs


vec2jac = JacobianEvaluator


class JacobianInvEvaluator(Evaluable):
  """Evaluator of Jacobian Inv."""

  def __init__(self, value: Evaluable) -> None:
    super().__init__()
    self._value = value

  @property
  def active(self) -> bool:
    return self._value.active

  def forward(self) -> Node:
    child = self._value.forward()
    value = se3op.vec2jacinv(child.value)
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    if self._value.active:
      lhs = 0.5 * lhs  # TODO: check this
      jacs = self._value.backward(lhs, node.children[0])
    return jacs


vec2jacinv = JacobianInvEvaluator