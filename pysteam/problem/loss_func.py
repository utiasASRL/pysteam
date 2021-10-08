import abc


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
