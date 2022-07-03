import numpy as np
from typing import List, Tuple

from pylgmath import Transformation

from ...evaluable.se3 import SE3StateVar
from ...evaluable.vspace import VSpaceStateVar
from ..time import Time
from .interface import Interface


class Interpolator:
  """A convenient class for trajectory interpolation that abstracts away the state variable and evaluator details in
  steam.
  """

  def __init__(self, *states: List[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    """
    Args:
      states (List[Tuple[float, np.ndarray, np.ndarray]]): list of states that is a tuple of
        (time (secs), T_k0, w_0k_in_k) where w_0k_in_k is body centric velocity.
    """
    self._trajectory: Interface = Interface()
    self.add_states(*states)

  def add_states(self, *states: List[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Adds states to the trajectory.
    Args:
      states (List[Tuple[float, np.ndarray, np.ndarray]]): list of states that is a tuple of
        (time (secs), T_k0, w_0k_in_k) where w_0k_in_k is body centric velocity.
    """
    for state in states:
      time = Time(secs=state[0])
      T_k0 = SE3StateVar(Transformation(T_ba=state[1]))
      w_0k_ink = VSpaceStateVar(state[2])
      self._trajectory.add_knot(time=time, T_k0=T_k0, w_0k_ink=w_0k_ink)

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
      T_k0 = self._trajectory.get_pose_interpolator(traj_time).evaluate().matrix()
      w_0k_ink = self._trajectory.get_velocity_interpolator(traj_time).evaluate()
      states.append([time, T_k0, w_0k_ink])
    return states[0] if len(times) == 1 else states
