from __future__ import annotations

import uuid
import abc
from copy import deepcopy


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

  def __init__(self, value, perturb_dim: int, *, is_locked: bool = False, copy: bool = False):
    assert perturb_dim > 0, "Zero or negative pertubation dimension."
    self._value = deepcopy(value) if copy else value
    self._perturb_dim: int = perturb_dim
    self._is_locked: bool = is_locked
    self._key: StateKey = StateKey()

  @abc.abstractmethod
  def update(self):
    pass

  @abc.abstractmethod
  def get_value(self):
    pass

  @abc.abstractmethod
  def set_value(self, value):
    pass

  @abc.abstractmethod
  def clone(other: StateVar):
    pass

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
