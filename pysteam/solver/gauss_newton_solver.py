import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spla_sparse

from . import Solver
from ..problem import OptimizationProblem


class GaussNewtonSolver(Solver):

  def __init__(self, problem: OptimizationProblem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update({
        "use_sparse_matrix": True,
    })
    self._parameters.update(**parameters)

    # for covariance query
    self._approx_hessian = None

  def linearize_solve_and_update(self):
    # build the system
    A, b = self.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check
    self._approx_hessian = A  # keep a copy of the LHS (i.e., the approximated Hessian)

    # solve the system
    perturbation = self.solve_gauss_newton(A, b)

    # apply update
    new_cost = self.propose_update(perturbation)
    self.accept_proposed_state()

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print("Iteration: {0:4}  -  Cost: {1:10.4f}".format(self._curr_iteration, new_cost))

    return True, new_cost, grad_norm

  def build_gauss_newton_terms(self) -> tuple:
    """Returns the LHS and RHS of the linear system: A, b."""
    return self._problem.build_gauss_newton_terms(self._state_vector, self._parameters["use_sparse_matrix"])

  def solve_gauss_newton(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the perturbation."""
    if self._parameters["use_sparse_matrix"]:
      return spla_sparse.spsolve(A, b)[..., None]  # expand to (state_size, 1)
    else:
      return spla.cho_solve(spla.cho_factor(A), b)

  def query_covariance(self, toarray=True):
    assert self._approx_hessian is not None
    # Hessian == inverse covariance
    if self._parameters["use_sparse_matrix"]:
      if toarray:
        return npla.inv(self._approx_hessian.toarray())
      else:
        return spla.inv(self._approx_hessian)
    else:
      return npla.inv(self._approx_hessian)