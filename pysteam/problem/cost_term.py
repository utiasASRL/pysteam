import abc
from typing import Dict, Tuple
import numpy as np
import numpy.linalg as npla

from ..state import StateKey, StateVector
from ..evaluator import Evaluator
from . import NoiseModel, LossFunc


class CostTerm(abc.ABC):
  """Interface for a 'cost term' class that contributes to the objective function."""

  @abc.abstractmethod
  def cost(self) -> float:
    """Computes the cost to the objective function"""

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

  def __init__(self, error_func: Evaluator, noise_model: NoiseModel, loss_func: LossFunc):
    super().__init__()
    self._error_func: Evaluator = error_func
    self._noise_model: NoiseModel = noise_model
    self._loss_func: LossFunc = loss_func

  def cost(self):
    error = self._error_func.evaluate()
    whitened_error = self._noise_model.get_whitened_error_norm(error)
    return self._loss_func.cost(whitened_error)

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

  def eval_weighted_and_whitened(self) -> Tuple[np.ndarray, Dict[StateKey, np.ndarray]]:
    """Computes the weighted and whitened errors and jacobians.
      err = sqrt(w)*sqrt(R^-1)*rawError
      jac = sqrt(w)*sqrt(R^-1)*rawJacobian
    """
    # get raw error and Jacobians
    raw_error, jacs = self._error_func.evaluate(self._noise_model.get_sqrt_information())
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