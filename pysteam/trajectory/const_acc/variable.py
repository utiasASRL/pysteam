from __future__ import annotations

from ...evaluable import Evaluable
from ..time import Time


class Variable:

  def __init__(self, time: Time, T_k0: Evaluable, w_0k_ink: Evaluable, dw_0k_ink: Evaluable) -> None:
    self._time: Time = time
    self._T_k0: Evaluable = T_k0
    self._w_0k_ink: Evaluable = w_0k_ink
    self._dw_0k_ink: Evaluable = dw_0k_ink
    assert self._w_0k_ink.perturb_dim == 6, "Invalid velocity size."
    assert self._dw_0k_ink.perturb_dim == 6, "Invalid acceleration size."

  @property
  def time(self) -> Time:
    return self._time

  @property
  def pose(self) -> Evaluable:
    return self._T_k0

  @property
  def velocity(self) -> Evaluable:
    return self._w_0k_ink

  @property
  def acceleration(self) -> Evaluable:
    return self._dw_0k_ink
