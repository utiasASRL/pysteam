from typing import List, Tuple
import array
import numpy as np
from scipy.sparse import coo_matrix, dok_matrix

from ..state import StateVar, StateVector
from . import CostTerm


class LazyCOOBuilder:

  class SlicingHandler:

    def __init__(self, key: Tuple[slice, slice]):
      self.key = key

      rows, cols = key
      rows = range(rows.stop)[rows]
      cols = range(cols.stop)[cols]
      self.rows = np.repeat(rows, len(cols))
      self.cols = np.tile(cols, len(rows))

      self.data = None

    def __iadd__(self, data: np.ndarray):
      self.data = data.reshape(-1)
      return self

  def __init__(self, shape: Tuple[int, int]):
    self.shape = shape
    self.rows = array.array('i')
    self.cols = array.array('i')
    self.data = array.array('d')

  def tocoo(self):
    return coo_matrix((self.data, (self.rows, self.cols)), self.shape)

  def __getitem__(self, key):
    return LazyCOOBuilder.SlicingHandler(key)

  def __setitem__(self, key: Tuple[slice, slice], value: SlicingHandler):
    assert key == value.key
    if value.data is not None:
      self.rows.extend(value.rows)
      self.cols.extend(value.cols)
      self.data.extend(value.data)


class OptimizationProblem:
  """Container for state variables and cost terms associated with the optimization problem to be solved."""

  def __init__(self) -> None:
    self._cost_terms: List[CostTerm] = []
    self._state_vars: List[StateVar] = []

  def add_state_var(self, *state_vars: StateVar) -> None:
    """Adds state variables (either locked or unlocked)."""
    self._state_vars.extend(state_vars)

  def add_cost_term(self, *cost_terms: CostTerm) -> None:
    """Adds cost terms."""
    self._cost_terms.extend(cost_terms)

  def cost(self) -> float:
    """Computes the cost from the collection of cost terms."""
    return sum([x.cost() for x in self._cost_terms])

  def get_state_vars(self) -> List[StateVar]:
    """Gets reference to state variables."""
    return self._state_vars

  def get_num_of_cost_terms(self) -> int:
    """Gets the total number of cost terms."""
    return len(self._cost_terms)

  def build_gauss_newton_terms(self, state_vector: StateVector, sparse: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the left-hand approximated Hessian (A) and right-hand gradient vector (b)."""
    state_size = state_vector.get_state_size()
    # equivalent to
    #   A = dok_matrix((state_size, state_size)) if sparse else np.zeros((state_size, state_size))
    #   ...
    #   return A.tocsr()
    # but faster
    A = LazyCOOBuilder((state_size, state_size)) if sparse else np.zeros((state_size, state_size))
    b = np.zeros((state_size, 1))
    for cost_term in self._cost_terms:
      cost_term.build_gauss_newton_terms(state_vector, A, b)
    return A.tocoo().tocsr() if sparse else A, b