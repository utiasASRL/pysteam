from __future__ import annotations

from ..state import VectorSpaceStateVar
from ..evaluator import TransformEvaluator


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

  def __init__(self, time: Time, T_k0: TransformEvaluator, velocity: VectorSpaceStateVar) -> None:
    self._time: Time = time
    self._T_k0: TransformEvaluator = T_k0
    self._velocity: VectorSpaceStateVar = velocity
    assert self._velocity.get_perturb_dim() == 6, "Invalid velocity size."

  @property
  def time(self) -> Time:
    return self._time

  @property
  def pose(self) -> TransformEvaluator:
    return self._T_k0

  @property
  def velocity(self) -> VectorSpaceStateVar:
    return self._velocity
