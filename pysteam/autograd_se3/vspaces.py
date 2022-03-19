import numpy as np
from ..autograd.extend import VSpace

from pylgmath import Transformation


class SE3VSpace(VSpace):

  def __init__(self, _):
    self.shape = (6,)
    self.dtype = np.float64

  @property
  def size(self): return np.prod(self.shape)
  @property
  def ndim(self): return len(self.shape)
  def zeros(self): return np.zeros(self.shape, dtype=self.dtype)
  def ones(self): return np.ones(self.shape, dtype=self.dtype)
  def standard_basis(self):
    for idxs in np.ndindex(*self.shape):
      vect = np.zeros(self.shape, dtype=self.dtype)
      vect[idxs] = 1
      yield vect
  def randn(self):
    return np.array(np.random.randn(*self.shape)).astype(self.dtype)


SE3VSpace.register(Transformation)
