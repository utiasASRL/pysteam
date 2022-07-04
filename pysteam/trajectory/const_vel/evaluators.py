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


class StateErrorEvaluator(Evaluable):
  """Evaluator for combined pose and velocity prior, for covariance interpolation"""

  def __init__(self, T_k0_error: Evaluable, w_0k_ink_error: Evaluable):
    super().__init__()
    self._T_k0_error: Evaluable = T_k0_error
    self._w_0k_ink_error: Evaluable = w_0k_ink_error

  @property
  def active(self) -> bool:
    return self._T_k0_error.active or self._w_0k_ink_error.active

  def forward(self) -> Node:
    T_k0_error = self._T_k0_error.forward()
    w_0k_ink_error = self._w_0k_ink_error.forward()

    value = np.concatenate((T_k0_error.value, w_0k_ink_error.value), axis=0)
    return Node(value, T_k0_error, w_0k_ink_error)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    T_k0_error, w_0k_ink_error = node.children

    if self._T_k0_error.active:
      self._T_k0_error.backward(lhs[..., :6], T_k0_error, jacs)

    if self._w_0k_ink_error.active:
      self._w_0k_ink_error.backward(lhs[..., 6:], w_0k_ink_error, jacs)


state_error = StateErrorEvaluator