from pylgmath import se3op, Transformation

from ..evaluable import Evaluable, Node
from .trajectory_var import Time


class ConstVelTransformEvaluator(Evaluable):
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

  def backward(self, lhs, node):
    jacs = dict()

    child = node.children[0]

    if self._velocity.active:
      xi = self._time.seconds * child.value
      jac = self._time.seconds * se3op.vec2jac(xi)
      jacs = self._velocity.backward(lhs @ jac, child)

    return jacs