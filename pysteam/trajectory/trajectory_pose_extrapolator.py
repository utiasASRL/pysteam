import numpy as np

from pylgmath import se3op, Transformation

from ..evaluable import Evaluable, Node, Jacobians
from .trajectory_var import Time


class PoseExtrapolator(Evaluable):
  """Simple transform evaluator for a constant velocity model."""

  def __init__(self, velocity: Evaluable, time: Time):
    super().__init__()
    self._velocity: Evaluable = velocity
    self._time: Time = time

  @property
  def active(self) -> bool:
    return self._velocity.active

  def forward(self) -> Node:
    child = self._velocity.forward()
    T_tk = Transformation(xi_ab=self._time.seconds * child.value)
    return Node(T_tk, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._velocity.active:
      child = node.children[0]
      xi = self._time.seconds * child.value
      jac = self._time.seconds * se3op.vec2jac(xi)
      self._velocity.backward(lhs @ jac, child, jacs)