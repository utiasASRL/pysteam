from __future__ import annotations

from ...evaluable import Evaluable
from ..time import Time


class Variable:

  def __init__(self, time: Time, c: Evaluable) -> None:
    self._time: Time = time
    self._c: Evaluable = c
    assert self._c.perturb_dim == 6, "Invalid c size."

  @property
  def time(self) -> Time:
    return self._time

  @property
  def c(self) -> Evaluable:
    return self._c