from typing import Dict
import numpy as np

from pylgmath import se3op

from ..evaluatable import StateKey
from ..evaluatable import Evaluatable, Node


class JinvVelocityEvaluator(Evaluatable):
  """Evaluator for the composition of transformation matrices."""

  def __init__(self, jacinv: Evaluatable, velocity: Evaluatable):
    super().__init__()
    self._jacinv: Evaluatable = jacinv
    self._velocity: Evaluatable = velocity

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