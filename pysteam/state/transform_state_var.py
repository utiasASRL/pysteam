import numpy as np

from pylgmath import se3op, Transformation
from .state_var import StateVar


class TransformStateVar(StateVar):
  """Variable wrapper of lgmath.Transformation."""

  def __init__(self, value: Transformation, **kwargs) -> None:
    super().__init__(value, 6, **kwargs)

  def get_value(self) -> Transformation:
    return self._value

  def set_value(self, value: Transformation) -> None:
    self._value.assign(T_ba=value.matrix())

  def update(self, perturbation: np.ndarray) -> None:
    self._value.assign(T_ba=se3op.vec2tran(perturbation) @ self._value.matrix())

  def clone(self):
    raise NotImplementedError