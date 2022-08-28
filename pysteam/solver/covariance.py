import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as spla

from ..evaluable import StateVar
from ..problem import Problem


class Covariance():

  def __init__(self, problem: Problem):
    self._state_vector = problem.get_state_vector()
    self._approx_hessian = problem.build_gauss_newton_terms()[0]

    # Hessian == inverse covariance
    if sp_sparse.issparse(self._approx_hessian):
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

    cov = self._covariance[np.ix_(rindices, cindices)]
    if sp_sparse.issparse(self._approx_hessian):
      cov = cov.toarray()
    return cov