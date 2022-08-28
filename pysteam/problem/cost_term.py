import abc
import numpy as np
import numpy.linalg as npla
from typing import Tuple

from ..evaluable import Evaluable, Jacobians
from .loss_func import LossFunc
from .noise_model import NoiseModel
from .state_vector import StateVector


class CostTerm(abc.ABC):
  """Interface for a 'cost term' class that contributes to the objective function."""

  def __init__(self, *, name: str = "") -> None:
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  @abc.abstractmethod
  def cost(self) -> float:
    """Computes the cost to the objective function"""

  @property
  @abc.abstractmethod
  def related_var_keys(self) -> set:
    """Returns the set of variables that are related to this cost term."""

  @abc.abstractmethod
  def build_gauss_newton_terms(self, state_vector: StateVector, A: np.ndarray, b: np.ndarray):
    """Add the contribution of this cost term to the left-hand (Hessian) and right-hand (gradient vector) sides of the
    Gauss-Newton system of equations.
    Args:
      state_vector (StateVector): vector of state variables that contains index information
      A (np.ndarray): left-hand Hessian
      b (np.ndarray): right-hand gradient vector
    """


class WeightedLeastSquareCostTerm(CostTerm):

  def __init__(self, error_func: Evaluable, noise_model: NoiseModel, loss_func: LossFunc, **kwargs) -> None:
    super().__init__(**kwargs)
    self._error_func: Evaluable = error_func
    self._noise_model: NoiseModel = noise_model
    self._loss_func: LossFunc = loss_func

  def cost(self):
    error = self._error_func.evaluate()
    whitened_error = self._noise_model.get_whitened_error_norm(error)
    return self._loss_func.cost(whitened_error)

  @property
  def related_var_keys(self) -> set:
    return self._error_func.related_var_keys

  def build_gauss_newton_terms(self, state_vector: StateVector, A: np.ndarray, b: np.ndarray) -> None:
    # compute the weighted and whitened errors and jacobians
    error, jacs = self.eval_weighted_and_whitened()

    # update the rhs
    for key, jac in jacs.items():
      idx = state_vector.get_state_indices(key)
      b[idx, :] += -1 * jac.T @ error

    # update the lhs
    for key1, jac1 in jacs.items():
      for key2, jac2 in jacs.items():
        idx1 = state_vector.get_state_indices(key1)
        idx2 = state_vector.get_state_indices(key2)
        A[idx1, idx2] += jac1.T @ jac2

  def eval_weighted_and_whitened(self) -> Tuple[np.ndarray, Jacobians]:
    """Computes the weighted and whitened errors and jacobians.
      err = sqrt(w)*sqrt(R^-1)*rawError
      jac = sqrt(w)*sqrt(R^-1)*rawJacobian
    """
    # get raw error and Jacobians
    jacs = Jacobians()
    raw_error = self._error_func.evaluate(self._noise_model.get_sqrt_information(), jacs)
    # get whitened error vector
    white_error = self._noise_model.whiten_error(raw_error)
    # get weight from loss function (IRLS)
    sqrt_w = np.sqrt(self._loss_func.weight(npla.norm(white_error)))

    # weight the white jacobians
    for key, jac in jacs.items():
      jacs[key] = sqrt_w * jac
    # weight the error
    error = sqrt_w * white_error

    return error, jacs