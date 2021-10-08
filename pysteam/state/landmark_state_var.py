import numpy as np
import numpy.linalg as npla

from .state_var import StateVar


class LandmarkStateVar(StateVar):
  """Variable wrapper of 4x1 numpy array as homogeneous coordinate of a landmark."""

  def __init__(self, value: np.ndarray, *, scale: bool = True, **kwargs) -> None:
    super().__init__(value, 3, **kwargs)
    assert value.shape == (4, 1)
    self._scale = scale
    self.refresh_homo_scalling()

  def get_value(self) -> np.ndarray:
    return self._value

  def set_value(self, value: np.ndarray) -> None:
    self._value[:] = value
    self.refresh_homo_scalling()

  def update(self, perturbation: np.ndarray) -> None:
    self._value += perturbation
    self.refresh_homo_scalling()

  def clone(self):
    raise NotImplementedError

  def refresh_homo_scalling(self) -> None:
    if self._scale:
      # get length of xyz-portion of landmark
      inv_mag = 1 / npla.norm(self._value[:3])
      # update to be unit length
      self._value *= inv_mag