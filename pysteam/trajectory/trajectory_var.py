from __future__ import annotations

from ..evaluable import Evaluable


class Time:

  def __init__(self, secs: float = 0, nsecs: int = 0) -> None:
    self._nsecs: int = int(secs * 1e9) + nsecs

  @property
  def seconds(self) -> float:
    return self._nsecs * 1e-9

  @property
  def nanosecs(self) -> int:
    return self._nsecs

  def __sub__(self, time: Time):
    return Time(nsecs=self.nanosecs - time.nanosecs)

  def __add__(self, time: Time):
    return Time(nsecs=self.nanosecs + time.nanosecs)


class TrajectoryVar:

  def __init__(self, time: Time, T_k0: Evaluable, w_0k_ink: Evaluable) -> None:
    self._time: Time = time
    self._T_k0: Evaluable = T_k0
    self._w_0k_ink: Evaluable = w_0k_ink
    assert self._w_0k_ink.perturb_dim == 6, "Invalid velocity size."

  @property
  def time(self) -> Time:
    return self._time

  @property
  def pose(self) -> Evaluable:
    return self._T_k0

  @property
  def velocity(self) -> Evaluable:
    return self._w_0k_ink
