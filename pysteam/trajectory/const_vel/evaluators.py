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

  @property
  def related_var_keys(self) -> set:
    return self._xi.related_var_keys | self._velocity.related_var_keys

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

  @property
  def related_var_keys(self) -> set:
    return self._xi.related_var_keys | self._velocity.related_var_keys

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

  @property
  def related_var_keys(self) -> set:
    return self._T_k0_error.related_var_keys | self._w_0k_ink_error.related_var_keys

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


class TwoStateErrorEvaluator(Evaluable):
  """Evaluator for combined pose and velocity prior, for covariance interpolation"""

  def __init__(self, knot1_error: Evaluable, knot2_error: Evaluable):
    super().__init__()
    self._knot1_error: Evaluable = knot1_error
    self._knot2_error: Evaluable = knot2_error

  @property
  def active(self) -> bool:
    return self._knot1_error.active or self._knot2_error.active

  @property
  def related_var_keys(self) -> set:
    return self._knot1_error.related_var_keys | self._knot2_error.related_var_keys

  def forward(self) -> Node:
    knot1_error = self._knot1_error.forward()
    knot2_error = self._knot2_error.forward()

    value = np.concatenate((knot1_error.value, knot2_error.value), axis=0)
    return Node(value, knot1_error, knot2_error)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    knot1_error, knot2_error = node.children

    if self._knot1_error.active:
      self._knot1_error.backward(lhs[..., :12], knot1_error, jacs)

    if self._knot2_error.active:
      self._knot2_error.backward(lhs[..., 12:], knot2_error, jacs)


two_state_error = TwoStateErrorEvaluator