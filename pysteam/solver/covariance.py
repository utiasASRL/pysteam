import numpy as np
import numpy.linalg as npla
import scipy.sparse.linalg as spla

from ..problem import OptimizationProblem, StateVector


class Covariance():

  def __init__(self, opt_prob: OptimizationProblem, use_sparse_matrix=True):
    self._problem = opt_prob
    self._state_vector = StateVector()
    for state_var in self._problem.get_state_vars():
      if not state_var.locked:
        self._state_vector.add_state_var(state_var)

    self._approx_hessian = self._problem.build_gauss_newton_terms(self._state_vector, use_sparse_matrix)[0]

    # Hessian == inverse covariance
    if use_sparse_matrix:
      self._covariance = spla.inv(self._approx_hessian.tocsc())
    else:
      self._covariance = npla.inv(self._approx_hessian)

  def query(self, rvar, cvar=None):
    if cvar is None:
      cvar = rvar
    rindices = self._state_vector.get_state_indices(rvar.key)
    cindices = self._state_vector.get_state_indices(cvar.key)
    return self._covariance[rindices, cindices].toarray()