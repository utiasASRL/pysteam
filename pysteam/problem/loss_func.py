import abc
import numpy as np


class LossFunc(abc.ABC):
  """Interface for a 'loss function' class."""

  @abc.abstractmethod
  def cost(self, whitened_error_norm: float) -> float:
    """Cost function (basic evaluation of the loss function)."""

  @abc.abstractmethod
  def weight(self, whitened_error_norm: float):
    """Weight for iteratively reweighted least-squares (influence function div. by error)."""


class L2LossFunc(LossFunc):

  def cost(self, whitened_error_norm: float) -> float:
    return 0.5 * whitened_error_norm * whitened_error_norm

  def weight(self, whitened_error_norm: float):
    return 1.0


class CauchyLossFunc(LossFunc):

  def __init__(self, k: float) -> None:
    super().__init__()
    self._k = k

  def cost(self, whitened_error_norm: float) -> float:
    e_div_k = np.abs(whitened_error_norm) / self._k
    return 0.5 * self._k * self._k * np.log(1.0 + e_div_k * e_div_k)

  def weight(self, whitened_error_norm: float):
    e_div_k = np.abs(whitened_error_norm) / self._k
    return 1.0 / (1.0 + e_div_k * e_div_k)


class DcsLossFunc(LossFunc):

  def __init__(self, k: float) -> None:
    super().__init__()
    self._k2 = k * k

  def cost(self, whitened_error_norm: float) -> float:
    e2 = whitened_error_norm * whitened_error_norm
    if (e2 <= self._k2):
      return 0.5 * e2
    else:
      return 2.0 * self._k2 * e2 / (self._k2 + e2) - 0.5 * self._k2

  def weight(self, whitened_error_norm: float):
    e2 = whitened_error_norm * whitened_error_norm
    if (e2 <= self._k2):
      return 1.0
    else:
      k2e2 = self._k2 + e2
      return 4.0 * self._k2 * self._k2 / (k2e2 * k2e2)


class GemanMcClureLossFunc(LossFunc):

  def __init__(self, k: float) -> None:
    super().__init__()
    self._k2 = k * k

  def cost(self, whitened_error_norm: float) -> float:
    e2 = whitened_error_norm * whitened_error_norm
    return 0.5 * e2 / (self._k2 + e2)

  def weight(self, whitened_error_norm: float):
    e2 = whitened_error_norm * whitened_error_norm
    k2e2 = self._k2 + e2
    return self._k2 * self._k2 / (k2e2 * k2e2)


# TODO: more robust loss functions