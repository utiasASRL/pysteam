import numpy as np
import numpy.linalg as npla

from ..state_var import StateVar
from ..evaluatable import Node


class HomoPointStateVar(StateVar):
  """Variable wrapper of a 4x1 homogeneous point."""

  def __init__(self, value: np.ndarray, *, scale: bool = True, **kwargs) -> None:
    super().__init__(3, **kwargs)
    assert value.shape == (4, 1)
    self._value = value
    self._scale = scale
    self.refresh_homo_scaling()

  def forward(self):
    return Node(self._value)

  def backward(self, lhs, node):
    return {self.key: lhs} if self.active else {}

  def clone(self):
    raise NotImplementedError

  @property
  def value(self) -> np.ndarray:
    return self._value

  def assign(self, value: np.ndarray) -> None:
    self._value[:] = value
    self.refresh_homo_scaling()

  def update(self, perturbation: np.ndarray) -> None:
    self.refresh_homo_scaling(True)
    self._value[:3] += perturbation

  def refresh_homo_scaling(self, force=False) -> None:
    if self._scale or force:
      # get length of xyz-portion of landmark
      inv_mag = 1 / npla.norm(self._value[:3])
      # update to be unit length
      self._value *= inv_mag