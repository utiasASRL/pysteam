from __future__ import annotations

import abc

from .state_key import StateKey
from .evaluable import Evaluable


class StateVar(Evaluable):
  """
  Representation of a state variable involved in the optimization problem, subclass defines what the state represents
  and the current value of the state. This class should serve as a wrapper of the variable so never create its own copy
  of the variable.
  """

  def __init__(self, perturb_dim: int, *, locked: bool = False, name: str = "") -> None:
    assert perturb_dim > 0, "Zero or negative perturbation dimension."
    self._perturb_dim: int = perturb_dim
    self._locked: bool = locked
    self._name: str = name
    self._key: StateKey = StateKey()

  @property
  def active(self) -> bool:
    """Evaluable interface"""
    return not self.locked

  @property
  def related_var_keys(self) -> set:
    return set() if self.locked else {self._key}

  @property
  @abc.abstractmethod
  def value(self):
    """Returns a reference to the internal value"""

  @abc.abstractmethod
  def assign(self, value):
    """Sets internal value from another internal value"""

  @abc.abstractmethod
  def update(self, perturbation):
    """Update value given a perturbation"""

  def set_from_copy(self, other: StateVar) -> None:
    assert self.key == other.key
    self.assign(other.value)

  @property
  def key(self) -> StateKey:
    return self._key

  @property
  def perturb_dim(self) -> int:
    return self._perturb_dim

  @property
  def locked(self) -> bool:
    return self._locked

  @locked.setter
  def locked(self, v: bool):
    self._locked = v

  @property
  def name(self) -> str:
    return self._name