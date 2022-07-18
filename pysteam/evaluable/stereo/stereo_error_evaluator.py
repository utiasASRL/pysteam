import numpy as np

from ..evaluable import Evaluable, Node, Jacobians


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


class StereoErrorEvaluator(Evaluable):
  """Stereo camera error function evaluator."""

  def __init__(self, meas: np.ndarray, intrinsics: CameraIntrinsics, landmark: Evaluable):
    super().__init__()

    self._meas: np.ndarray = meas
    self._intrinsics: CameraIntrinsics = intrinsics
    self._landmark: Evaluable = landmark

  @property
  def active(self) -> bool:
    return self._landmark.active

  @property
  def related_var_keys(self) -> set:
    return self._landmark.related_var_keys

  def forward(self) -> Node:
    child = self._landmark.forward()

    point_in_cam_frame: np.ndarray = child.value
    error = self._meas - camera_model(self._intrinsics, point_in_cam_frame)

    return Node(error, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._landmark.active:
      child = node.children[0]

      point_in_cam_frame: np.ndarray = child.value
      lhs = -lhs @ camera_model_jac(self._intrinsics, point_in_cam_frame)

      self._landmark.backward(lhs, child, jacs)
