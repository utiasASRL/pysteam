from __future__ import annotations

import uuid
import abc


class StateKey:

  def __init__(self):
    self.id: uuid.UUID = uuid.uuid4()

  def __eq__(self, other: StateKey):
    return self.id == other.id

  def __ne__(self, other: StateKey):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.id)


class StateVar(abc.ABC):
  """
  Representation of a state variable involved in the optimization problem, subclass defines what the state represents
  and the current value of the state. This class should serve as a wrapper of the variable so never create its own copy
  of the variable.
  """

  def __init__(self, perturb_dim: int, *, is_locked: bool = False):
    assert perturb_dim > 0, "Zero or negative perturbation dimension."
    self._perturb_dim: int = perturb_dim
    self._is_locked: bool = is_locked
    self._key: StateKey = StateKey()

  @abc.abstractmethod
  def clone(other: StateVar):
    """Creats a copy of this state variable including its current value"""

  @abc.abstractmethod
  def get_value(self):
    """Returns a reference to the internal value"""

  @abc.abstractmethod
  def set_value(self, value):
    """Sets internal value from another internal value"""

  @abc.abstractmethod
  def update(self, perturbation):
    """Update value given a perturbation"""

  def set_from_copy(self, other: StateVar) -> None:
    assert self.get_key() == other.get_key()
    self.set_value(other.get_value())

  def get_key(self) -> StateKey:
    return self._key

  def get_perturb_dim(self) -> int:
    return self._perturb_dim

  def is_locked(self) -> bool:
    return self._is_locked

  def set_lock(self, lock_state: bool) -> None:
    self._is_locked = lock_state
