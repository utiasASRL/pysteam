import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as spla_sparse

from ..problem import Problem
from .solver import Solver


class GaussNewtonSolver(Solver):

  def __init__(self, problem: Problem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update(**parameters)

  def linearize_solve_and_update(self):
    # build the system
    A, b = self._problem.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check

    # solve the system
    perturbation = self.solve_gauss_newton(A, b)

    # apply update
    new_cost = self.propose_update(perturbation)
    self.accept_proposed_state()

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print(f"Iteration: {self._curr_iteration:4}  -  Cost: {new_cost:10.4f}")

    return True, new_cost, grad_norm

  def solve_gauss_newton(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the perturbation."""
    if sp_sparse.issparse(A):
      return spla_sparse.spsolve(A, b)[..., None]  # expand to (state_size, 1)
    else:
      return spla.cho_solve(spla.cho_factor(A), b)
