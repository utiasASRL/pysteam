import numpy as np

from pylgmath import se3op, Transformation

from ..evaluable import Evaluable, Node, Jacobians


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

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._value.active:
      lhs = lhs @ se3op.vec2jac(node.value.vec())
      self._value.backward(lhs, node.children[0], jacs)


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

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._transform.active:
      lhs = lhs @ se3op.vec2jacinv(node.value)
      self._transform.backward(lhs, node.children[0], jacs)


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

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._transform.active:
      lhs = -lhs @ node.value.adjoint()
      self._transform.backward(lhs, node.children[0], jacs)


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

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._transform1.active:
      self._transform1.backward(lhs, node.children[0], jacs)
    if self._transform2.active:
      lhs = lhs @ node.children[0].value.adjoint()
      self._transform2.backward(lhs, node.children[1], jacs)


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

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._transform1.active:
      self._transform1.backward(lhs, node.children[0], jacs)
    if self._transform2.active:
      tf_ba = node.children[0].value @ node.children[1].value.inverse()
      lhs = -lhs @ tf_ba.adjoint()
      self._transform2.backward(lhs, node.children[1], jacs)


compose_rinv = ComposeInverseEvaluator


class ComposeVelocityEvaluator(Evaluable):
  """Evaluator for the composition of two transformation matrices (with one inverted)."""

  def __init__(self, transform, velocity):
    super().__init__()
    self._transform = transform
    self._velocity = velocity

  @property
  def active(self) -> bool:
    return self._transform.active or self._velocity.active

  def forward(self):
    child1 = self._transform.forward()
    child2 = self._velocity.forward()

    value = se3op.tranAd(child1.value.matrix()) @ child2.value
    return Node(value, child1, child2)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._transform.active:
      jac = -se3op.curlyhat(node.value)
      # print(jac)
      self._transform.backward(lhs @ jac, node.children[0], jacs)
    if self._velocity.active:
      jac = se3op.tranAd(node.children[0].value.matrix())
      self._velocity.backward(lhs @ jac, node.children[1], jacs)


compose_velocity = ComposeVelocityEvaluator
