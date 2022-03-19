from typing import Dict
import numpy as np

from pylgmath import se3op

from ..evaluable import StateKey
from ..evaluable import Evaluable, Node


class JVelocityEvaluator(Evaluable):
  """Evaluator for the composition of left Jacobian and velocity."""

  def __init__(self, left_jac: Evaluable, velocity: Evaluable):
    super().__init__()
    self._left_jac: Evaluable = left_jac
    self._velocity: Evaluable = velocity

  @property
  def active(self) -> bool:
    return self._left_jac.active or self._velocity.active

  def forward(self) -> Node:
    left_jac = self._left_jac.forward()
    velocity = self._velocity.forward()

    value = left_jac.value @ velocity.value
    return Node(value, left_jac, velocity)

  def backward(self, lhs, node):
    print("WARNING: JVelocityEvaluator.backward computation may be incorrect.")
    jacs = dict()

    left_jac, velocity = node.children

    if self._left_jac.active:
      jacs = self._left_jac.backward(lhs @ se3op.curlyhat(velocity.value), left_jac)

    if self._velocity.active:
      jacs2 = self._velocity.backward(lhs @ left_jac.value, velocity)
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


j_velocity = JVelocityEvaluator


class JinvVelocityEvaluator(Evaluable):
  """Evaluator for the composition of Jinv and Velocity."""

  def __init__(self, jacinv: Evaluable, velocity: Evaluable):
    super().__init__()
    self._jacinv: Evaluable = jacinv
    self._velocity: Evaluable = velocity

  @property
  def active(self) -> bool:
    return self._jacinv.active or self._velocity.active

  def forward(self) -> Node:
    jacinv = self._jacinv.forward()
    velocity = self._velocity.forward()

    value = jacinv.value @ velocity.value
    return Node(value, jacinv, velocity)

  def backward(self, lhs, node):
    jacs = dict()

    jacinv, velocity = node.children

    if self._jacinv.active:
      jacs = self._jacinv.backward(lhs @ se3op.curlyhat(velocity.value), jacinv)

    if self._velocity.active:
      jacs2 = self._velocity.backward(lhs @ jacinv.value, velocity)
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


jinv_velocity = JinvVelocityEvaluator