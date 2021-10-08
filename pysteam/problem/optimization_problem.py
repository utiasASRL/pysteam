from typing import List, Tuple

import numpy as np

from ..state import StateVar, StateVector
from . import CostTerm


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

  def build_gauss_newton_terms(self, state_vector: StateVector) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the left-hand approximated Hessian (A) and right-hand gradient vector (b)."""
    state_size = state_vector.get_state_size()
    A = np.zeros((state_size, state_size))
    b = np.zeros((state_size, 1))
    for cost_term in self._cost_terms:
      cost_term.build_gauss_newton_terms(state_vector, A, b)
    return A, b