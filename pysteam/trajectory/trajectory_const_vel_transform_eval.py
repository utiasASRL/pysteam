import numpy as np

from pylgmath import se3op, Transformation

from ..state import VectorSpaceStateVar
from ..evaluator import EvalTreeNode, TransformEvaluator
from . import Time


class ConstVelTransformEvaluator(TransformEvaluator):
  """Simple transform evaluator for a constant velocity model."""

  def __init__(self, velocity: VectorSpaceStateVar, time: Time):
    super().__init__()
    self._velocity: VectorSpaceStateVar = velocity
    self._time: Time = time

  def is_active(self):
    return not self._velocity.is_locked()

  def get_eval_tree(self):
    xi = self._time.seconds * self._velocity.get_value()
    T_tk = Transformation(xi_ab=xi)
    return EvalTreeNode(T_tk)

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if not self._velocity.is_locked():
      xi = self._time.seconds * self._velocity.get_value()
      jac = self._time.seconds * se3op.vec2jac(xi)
      jacs = {self._velocity.get_key(): lhs @ jac}

    return jacs