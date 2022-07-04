import numpy as np
import numpy.linalg as npla
import scipy.sparse.linalg as spla

from ..evaluable import StateVar
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

  def query(self, rvars: StateVar, cvars: StateVar = None):
    if isinstance(rvars, StateVar):
      rvars = [rvars]
    if isinstance(cvars, StateVar):
      cvars = [cvars]
    if cvars is None:
      cvars = rvars
    rindices_slices = [self._state_vector.get_state_indices(rvar.key) for rvar in rvars]
    cindices_slices = [self._state_vector.get_state_indices(cvar.key) for cvar in cvars]
    rindices = [i for s in rindices_slices for i in range(s.start, s.stop)]
    cindices = [i for s in cindices_slices for i in range(s.start, s.stop)]
    return self._covariance[np.ix_(rindices, cindices)].toarray()