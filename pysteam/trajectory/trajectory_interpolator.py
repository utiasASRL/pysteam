import numpy as np
from typing import List, Tuple

from pylgmath import Transformation
from . import Time, TrajectoryInterface
from ..state import TransformStateVar, VectorSpaceStateVar
from ..evaluator import TransformStateEvaluator


class TrajectoryInterpolator:
  """A convenient class for trajectory interpolation that abstracts away the state variable and evaluator details in
  steam.
  """

  def __init__(self, *states: List[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    """
    Args:
      states (List[Tuple[float, np.ndarray, np.ndarray]]): list of states that is a tuple of
        (time (secs), T_k0, w_0k_in_k) where w_0k_in_k is body centric velocity.
    """
    self._trajectory: TrajectoryInterface = TrajectoryInterface(allow_extrapolation=False)
    self.add_states(*states)

  def add_states(self, *states: List[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Adds states to the trajectory.
    Args:
      states (List[Tuple[float, np.ndarray, np.ndarray]]): list of states that is a tuple of
        (time (secs), T_k0, w_0k_in_k) where w_0k_in_k is body centric velocity.
    """
    for state in states:
      time = Time(secs=state[0])
      pose = TransformStateEvaluator(TransformStateVar(Transformation(T_ba=state[1])))
      velocity = VectorSpaceStateVar(state[2])
      self._trajectory.add_knot(time=time, T_k0=pose, velocity=velocity)

  def get_states(self, *times: List[float]):
    """Gets interpolated states at specified time stamps.
    Args:
      times (List[float]): list of query time stamps in seconds.
    Returns:
      List[Tuple[float, np.ndarray, np.ndarray]]: interpolated states at specified time stamps
    """
    states = []
    for time in times:
      traj_time = Time(secs=time)
      pose = self._trajectory.get_interp_pose_eval(traj_time).evaluate().matrix()
      velocity = self._trajectory.get_interp_velocity(traj_time)
      states.append([time, pose, velocity])
    return states[0] if len(times) == 1 else states
