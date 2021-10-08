from typing import Optional
import numpy as np

from ..state import LandmarkStateVar
from . import EvalTreeNode, Evaluator, TransformEvaluator, ComposeLandmarkEvaluator


class CameraIntrinsics:

  def __init__(self, fu: float, fv: float, cu: float, cv: float, b: float) -> None:
    """Simple class to hold the stereo camera intrinsics.
    Args:
      fu (float): Focal length in the u-coordinate (horizontal)
      fv (float): Focal length in the v-coordinate (vertical)
      cu (float): Focal center offset in the u-coordinate (horizontal)
      cv (float): Focal center offset in the v-coordinate (vertical)
      b (float): Stereo baseline
    """
    self.fu = fu
    self.fv = fv
    self.cu = cu
    self.cv = cv
    self.b = b


def camera_model(ints: CameraIntrinsics, p: np.ndarray) -> np.ndarray:
  g = np.zeros(p.shape[:-2] + (4, 1))
  g[..., 0, 0] = ints.fu * p[..., 0, 0] / p[..., 2, 0] + ints.cu
  g[..., 1, 0] = ints.fv * p[..., 1, 0] / p[..., 2, 0] + ints.cv
  g[..., 2, 0] = ints.fu * (p[..., 0, 0] - p[..., 3, 0] * ints.b) / p[..., 2, 0] + ints.cu
  g[..., 3, 0] = ints.fv * p[..., 1, 0] / p[..., 2, 0] + ints.cv
  return g


def camera_model_jac(ints: CameraIntrinsics, p: np.ndarray) -> np.ndarray:
  dgdp = np.zeros(p.shape[:-2] + (4, 4))
  dgdp[..., 0, 0] = ints.fu / p[..., 2, 0]
  dgdp[..., 0, 2] = -ints.fu * p[..., 0, 0] / (p[..., 2, 0]**2)
  dgdp[..., 1, 1] = ints.fv / p[..., 2, 0]
  dgdp[..., 1, 2] = -ints.fv * p[..., 1, 0] / (p[..., 2, 0]**2)
  dgdp[..., 2, 0] = ints.fu / p[..., 2, 0]
  dgdp[..., 2, 2] = -ints.fu * (p[..., 0, 0] - p[..., 3, 0] * ints.b) / (p[..., 2, 0]**2)
  dgdp[..., 2, 3] = -ints.fu * ints.b / p[..., 2, 0]
  dgdp[..., 3, 1] = ints.fv / p[..., 2, 0]
  dgdp[..., 3, 2] = -ints.fv * p[..., 1, 0] / (p[..., 2, 0]**2)
  return dgdp


class StereoCameraErrorEval(Evaluator):
  """Stereo camera error function evaluator."""

  def __init__(self, meas: np.ndarray, intrinsics: CameraIntrinsics, T_cam_landmark: TransformEvaluator,
               landmark: LandmarkStateVar):
    super().__init__()

    self._meas: np.ndarray = meas
    self._intrinsics: CameraIntrinsics = intrinsics
    self._error_eval: TransformEvaluator = ComposeLandmarkEvaluator(T_cam_landmark, landmark)

  def is_active(self):
    return self._error_eval.is_active()

  def evaluate(self, lhs: Optional[np.ndarray] = None):

    tree: EvalTreeNode = self._error_eval.get_eval_tree()
    point_in_cam_frame: np.ndarray = tree.value

    error = self._meas - camera_model(self._intrinsics, point_in_cam_frame)

    if lhs is None:
      return error

    lhs = -lhs @ camera_model_jac(self._intrinsics, point_in_cam_frame)
    jacs = self._error_eval.compute_jacs(lhs, tree)

    return error, jacs