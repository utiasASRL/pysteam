import numpy as np

from pylgmath import se3op

from ...evaluable import Evaluable, Node, Jacobians


class JVelocityEvaluator(Evaluable):
  """Evaluator for the composition of left Jacobian and velocity."""

  def __init__(self, xi: Evaluable, velocity: Evaluable):
    super().__init__()
    self._xi: Evaluable = xi
    self._velocity: Evaluable = velocity

  @property
  def active(self) -> bool:
    return self._xi.active or self._velocity.active

  def forward(self) -> Node:
    xi = self._xi.forward()
    velocity = self._velocity.forward()

    value = se3op.vec2jac(xi.value) @ velocity.value
    return Node(value, xi, velocity)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    xi, velocity = node.children

    if self._xi.active:
      self._xi.backward(-0.5 * lhs @ se3op.curlyhat(velocity.value) @ se3op.vec2jac(xi.value), xi, jacs)

    if self._velocity.active:
      self._velocity.backward(lhs @ se3op.vec2jac(xi.value), velocity, jacs)


j_velocity = JVelocityEvaluator


class JinvVelocityEvaluator(Evaluable):
  """Evaluator for the composition of Jinv and Velocity."""

  def __init__(self, xi: Evaluable, velocity: Evaluable):
    super().__init__()
    self._xi: Evaluable = xi
    self._velocity: Evaluable = velocity

  @property
  def active(self) -> bool:
    return self._xi.active or self._velocity.active

  def forward(self) -> Node:
    xi = self._xi.forward()
    velocity = self._velocity.forward()

    value = se3op.vec2jacinv(xi.value) @ velocity.value
    return Node(value, xi, velocity)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    xi, velocity = node.children

    if self._xi.active:
      self._xi.backward(0.5 * lhs @ se3op.curlyhat(velocity.value) @ se3op.vec2jacinv(xi.value), xi, jacs)

    if self._velocity.active:
      self._velocity.backward(lhs @ se3op.vec2jacinv(xi.value), velocity, jacs)


jinv_velocity = JinvVelocityEvaluator