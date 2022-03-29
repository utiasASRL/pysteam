import numpy as np

from ..state_var import StateVar
from ..evaluable import Node, Jacobians


class VSpaceStateVar(StateVar):
  """Variable wrapper of a nx1 numpy array."""

  def __init__(self, value: np.ndarray, **kwargs) -> None:
    assert len(value.shape) == 2 and value.shape[-1] == 1
    super().__init__(value.shape[-2], **kwargs)
    self._value = value

  def forward(self):
    return Node(self._value)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self.active:
      jacs.add(self.key, lhs)

  @property
  def value(self) -> np.ndarray:
    return self._value

  def assign(self, value: np.ndarray) -> None:
    self._value[:] = value

  def update(self, perturbation: np.ndarray) -> None:
    self._value += perturbation
