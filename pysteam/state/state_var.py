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

  def __init__(self, perturb_dim: int, *, locked: bool = False):
    assert perturb_dim > 0, "Zero or negative perturbation dimension."
    self._perturb_dim: int = perturb_dim
    self._locked: bool = locked
    self._key: StateKey = StateKey()

  @abc.abstractmethod
  def clone(other: StateVar):
    """Creats a copy of this state variable including its current value"""

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