import abc
import numpy as np
import numpy.linalg as npla


class NoiseEvaluator(abc.ABC):

  @abc.abstractmethod
  def evaluate(self) -> np.ndarray:
    """Evaluates the uncertainty based on a derived model."""


class NoiseModel(abc.ABC):

  def __init__(self, matrix: np.ndarray, type: str = "covariance"):
    if type == "covariance":
      self.set_by_covariance(matrix)
    elif type == "information":
      self.set_by_information(matrix)
    elif type == "sqrt_information":
      self.set_by_sqrt_information(matrix)
    else:
      raise RuntimeError("Unknown matrix type.")

  def set_by_covariance(self, matrix: np.ndarray) -> None:
    self.set_by_information(npla.inv(matrix))

  def set_by_information(self, matrix: np.ndarray) -> None:
    assert np.all(npla.eigvals(matrix) > 0)
    self.set_by_sqrt_information(npla.cholesky(matrix).T)

  def set_by_sqrt_information(self, matrix: np.ndarray) -> None:
    self._sqrt_information = matrix  # stores the upper triangular part

  @abc.abstractmethod
  def get_sqrt_information(self):
    """Gets a reference to the square root information matrix."""

  @abc.abstractmethod
  def get_whitened_error_norm(self, raw_error) -> float:
    """Gets the norm of the whitened error vector, sqrt(rawError^T * info * rawError)."""

  @abc.abstractmethod
  def whiten_error(self, raw_error: np.ndarray) -> np.ndarray:
    """Gets the whitened error vector, sqrtInformation*rawError."""


class StaticNoiseModel(NoiseModel):

  def get_sqrt_information(self) -> np.ndarray:
    return self._sqrt_information

  def whiten_error(self, raw_error: np.ndarray) -> np.ndarray:
    return self._sqrt_information @ raw_error

  def get_whitened_error_norm(self, raw_error: np.ndarray) -> float:
    return npla.norm(self.whiten_error(raw_error))


class DynamicNoiseModel(NoiseModel):

  def __init__(self, evaluator: NoiseEvaluator) -> None:
    super().__init__(evaluator.evaluate())
    self._evaluator = evaluator

  def get_sqrt_information(self):
    self.set_by_covariance(self._evaluator.evaluate())
    return self._sqrt_information

  def get_whitened_error_norm(self, raw_error):
    self.set_by_covariance(self._evaluator.evaluate())
    return npla.norm(self._sqrt_information @ raw_error)

  def whiten_error(self, raw_error: np.ndarray) -> np.ndarray:
    self.set_by_covariance(self._evaluator.evaluate())
    return self._sqrt_information @ raw_error
