from typing import Dict
import numpy as np

from pylgmath import se3op

from ..evaluable import StateKey
from ..evaluable import Evaluable, Node


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

  def backward(self, lhs, node):
    jacs = dict()

    xi, velocity = node.children

    if self._xi.active:
      jacs = self._xi.backward(-0.5 * lhs @ se3op.curlyhat(velocity.value) @ se3op.vec2jac(xi.value), xi)

    if self._velocity.active:
      jacs2 = self._velocity.backward(lhs @ se3op.vec2jac(xi.value), velocity)
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


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

  def backward(self, lhs, node):
    jacs = dict()

    xi, velocity = node.children

    if self._xi.active:
      jacs = self._xi.backward(0.5 * lhs @ se3op.curlyhat(velocity.value) @ se3op.vec2jacinv(xi.value), xi)

    if self._velocity.active:
      jacs2 = self._velocity.backward(lhs @ se3op.vec2jacinv(xi.value), velocity)
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


jinv_velocity = JinvVelocityEvaluator