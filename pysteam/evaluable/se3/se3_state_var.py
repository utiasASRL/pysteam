import numpy as np

from pylgmath import se3op, Transformation

from ..state_var import StateVar
from ..evaluable import Node, Jacobians


class SE3StateVar(StateVar):
  """Variable wrapper of lgmath.Transformation."""

  def __init__(self, value: Transformation, **kwargs) -> None:
    super().__init__(6, **kwargs)
    self._value = value

  def forward(self):
    return Node(self._value)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self.active:
      jacs.add(self.key, lhs)

  @property
  def value(self) -> Transformation:
    return self._value

  def assign(self, value: Transformation) -> None:
    self._value.assign(T_ba=value.matrix())

  def update(self, perturbation: np.ndarray) -> None:
    self._value.assign(T_ba=se3op.vec2tran(perturbation) @ self._value.matrix())
