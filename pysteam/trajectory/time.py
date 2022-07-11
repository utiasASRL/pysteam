from __future__ import annotations


class Time:

  def __init__(self, secs: float = 0, nsecs: int = 0) -> None:
    self._nsecs: int = int(secs * 1e9) + nsecs

  @property
  def seconds(self) -> float:
    return self._nsecs * 1e-9

  @property
  def nanosecs(self) -> int:
    return self._nsecs

  def __eq__(self, other):
    return self._nsecs == other._nsecs

  def __hash__(self) -> int:
    return hash(self._nsecs)

  def __sub__(self, time: Time):
    return Time(nsecs=self.nanosecs - time.nanosecs)

  def __add__(self, time: Time):
    return Time(nsecs=self.nanosecs + time.nanosecs)