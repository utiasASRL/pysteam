from ..autograd.extend import Box

from pylgmath import Transformation


class SE3Box(Box):
  __slots__ = []

  # @primitive
  # def __getitem__(A, idx): return A[idx]

  # Constants w.r.t float data just pass though
  # shape = property(lambda self: self._value.shape)
  # ndim  = property(lambda self: self._value.ndim)
  # size  = property(lambda self: self._value.size)
  # dtype = property(lambda self: self._value.dtype)
  # def __len__(self): return len(self._value)
  def __hash__(self): return id(self)


SE3Box.register(Transformation)
