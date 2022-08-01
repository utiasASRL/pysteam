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


class ComposeCurlyhatEvaluator(Evaluable):
  """Evaluator for the composition curlyhat(x) @ y."""

  def __init__(self, x: Evaluable, y: Evaluable):
    super().__init__()
    self._x: Evaluable = x
    self._y: Evaluable = y

  @property
  def active(self) -> bool:
    return self._x.active or self._y.active

  @property
  def related_var_keys(self) -> set:
    return self._x.related_var_keys | self._y.related_var_keys

  def forward(self) -> Node:
    x = self._x.forward()
    y = self._y.forward()
    value = se3op.curlyhat(x.value) @ y.value
    return Node(value, x, y)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    x, y = node.children

    if self._x.active:
      self._x.backward(-lhs @ se3op.curlyhat(y.value), x, jacs)

    if self._y.active:
      self._y.backward(lhs @ se3op.curlyhat(x.value), y, jacs)


compose_curlyhat = ComposeCurlyhatEvaluator