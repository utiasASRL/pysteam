from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..state import StateKey, StateVar


class StateContainer:

  def __init__(self, state_var: StateVar, indices: List[int]):
    self.state_var: StateVar = state_var
    self.indices: List[int] = indices


class StateVector:

  def __init__(self) -> None:
    self._state_vars: Dict[StateKey, StateContainer] = dict()
    self._state_indices = [0]
    self._state_size = 0

  def add_state_var(self, state_var: StateVar) -> None:
    """Add state variable."""
    if state_var.is_locked():
      raise RuntimeError("Cannot add locked state variable to an optimizable state vector.")

    key = state_var.get_key()
    if self.has_state_var(key):
      raise RuntimeError("StateVector already contains the state being added.")

    self._state_size += state_var.get_perturb_dim()
    self._state_indices.append(self._state_indices[-1] + state_var.get_perturb_dim())
    indices = slice(self._state_indices[-2], self._state_indices[-1])
    self._state_vars[key] = StateContainer(state_var, indices)

  def has_state_var(self, key: StateKey):
    """Check if a state variable exists in the vector."""
    return key in self._state_vars

  def get_state_var(self, key: StateKey) -> StateVar:
    """Get a state variable using a key."""
    assert key in self._state_vars
    return self._state_vars[key].state_var

  def get_num_states(self) -> int:
    """Get number of state variables."""
    return len(self._state_vars)

  def get_state_indices(self, key: StateKey) -> List[int]:
    """Get the block index of a state."""
    return self._state_vars[key].indices

  def get_state_size(self) -> int:
    """Get the total size of the states (i.e. sum of dimension of state variables)."""
    return self._state_size

  def update(self, perturbation: np.ndarray) -> None:
    """Update the state vector."""
    for state in self._state_vars.values():
      state.state_var.update(perturbation[state.indices])

  def copy_values(self, other: StateVector) -> None:
    """Copy the values of 'other' into 'this' (states must already align, typically this means that one is already a
    deep copy of the other)"""
    assert self.get_num_states() == other.get_num_states()
    for k, v in self._state_vars.items():
      v.state_var.set_from_copy(other.get_state_var(k))
