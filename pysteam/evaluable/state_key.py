from __future__ import annotations

import uuid


class StateKey:

  def __init__(self):
    self.id: uuid.UUID = uuid.uuid4()

  def __eq__(self, other: StateKey):
    return self.id == other.id

  def __ne__(self, other: StateKey):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.id)