"""
Examples that align two point clouds with known correspondences - one uses trajectory interpolation to handle motion
distortion.
"""
import numpy as np
import matplotlib.pyplot as plt

from pylgmath import cmnop, se3op, Transformation
from pysteam.evaluator import TransformStateEvaluator, InverseTransformEvaluator, PointToPointErrorEval
from pysteam.state import TransformStateVar, VectorSpaceStateVar
from pysteam.trajectory import Time, TrajectoryInterface
from pysteam.problem import L2LossFunc, StaticNoiseModel, WeightedLeastSquareCostTerm, OptimizationProblem
from pysteam.solver import GaussNewtonSolver

np.set_printoptions(linewidth=160, precision=4)


def without_motion_correction(fig=None):
  ## Create a 360 degree scene with 90 degree vertical FOV
  # horizontal and vertical resolution
  hori_res = np.pi / 6
  vert_res = np.pi / 24
  # horizontal and vertical degrees in radian
  phis = np.arange(0, 2 * np.pi, hori_res)
  thetas = np.arange(0, np.pi / 2 + 1e-5, vert_res)
  # tile and repeat to create lidar scans
  fullphis = np.repeat(phis, len(thetas))
  fullthetas = np.tile(thetas, len(phis))
  fullrhos = np.random.uniform(10, 10, fullthetas.shape)
  # create reference point cloud
  ref_pts_polar = np.stack((fullrhos, fullthetas, fullphis), axis=-1)
  ref_pts_polar = ref_pts_polar[..., None]
  ref_pts_cart = cmnop.pol2cart(ref_pts_polar)
  ref_points = np.concatenate((ref_pts_cart, np.ones((ref_pts_cart.shape[0], 1, 1))), axis=-2)

  ## Create query point cloud (no motion distortion)
  # ground truth transformation
  gt_T_10_vec = np.array([[-2.2, 0., 0., 0., 0., 0.44]]).T
  gt_T_10 = se3op.vec2tran(gt_T_10_vec)
  # transform point cloud
  qry_points = gt_T_10 @ ref_points
  # correspondences
  sample_inds = np.array([np.arange(qry_points.shape[0]), np.arange(qry_points.shape[0])]).T

  ## Setup states
  T_10_var = TransformStateVar(Transformation())
  T_10_eval = TransformStateEvaluator(T_10_var)

  ## Setup shared noise and loss function
  noise_model = StaticNoiseModel(np.eye(3))
  loss_func = L2LossFunc()

  ## Setup cost terms
  cost_terms = []
  for i in range(qry_points.shape[0]):
    qry_pt = qry_points[sample_inds[i, 0]]
    ref_pt = ref_points[sample_inds[i, 1]]
    error_func = PointToPointErrorEval(InverseTransformEvaluator(T_10_eval), ref_pt, qry_pt)
    cost_terms.append(WeightedLeastSquareCostTerm(error_func, noise_model, loss_func))

  ## Make optimization problem
  opt_prob = OptimizationProblem()
  opt_prob.add_state_var(T_10_var)
  opt_prob.add_cost_term(*cost_terms)

  ## Make solver and solve
  optimizer = GaussNewtonSolver(opt_prob, verbose=True)
  optimizer.optimize()

  print("Estimated Transform:     \n", T_10_var.get_value())
  print("Ground Truth Transform:  \n", gt_T_10)
  assert np.allclose(T_10_var.get_value().matrix(), gt_T_10)

  ## Get interpolated poses to align point cloud
  aligned = T_10_var.get_value().inverse().matrix() @ qry_points

  ## Plot if figure is given
  if fig is not None:
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Point Cloud Alignment without Motion Distortion')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(ref_points[:, 0, 0], ref_points[:, 1, 0], ref_points[:, 2, 0], s=10, c='g', label='Reference Points')
    ax.scatter(qry_points[:, 0, 0],
               qry_points[:, 1, 0],
               qry_points[:, 2, 0],
               s=10,
               c='r',
               label='Query Points (Motion Distorted)')
    ax.scatter(aligned[:, 0, 0], aligned[:, 1, 0], aligned[:, 2, 0], s=10, c='b', label='Aligned Points')
    ax.legend()


def with_motion_correction(fig=None):
  ## Create a 360 degree scene with 90 degree vertical FOV
  # horizontal and vertical resolution
  hori_res = np.pi / 6
  vert_res = np.pi / 24
  # horizontal and vertical degrees in radian
  phis = np.arange(0, 2 * np.pi, hori_res)
  thetas = np.arange(0, np.pi / 2 + 1e-5, vert_res)
  # tile and repeat to create lidar scans
  fullphis = np.repeat(phis, len(thetas))
  fullthetas = np.tile(thetas, len(phis))
  fullrhos = np.random.uniform(10, 10, fullthetas.shape)
  # create reference point cloud
  ref_pts_polar = np.stack((fullrhos, fullthetas, fullphis), axis=-1)
  ref_pts_polar = ref_pts_polar[..., None]
  ref_pts_cart = cmnop.pol2cart(ref_pts_polar)
  ref_points = np.concatenate((ref_pts_cart, np.ones((ref_pts_cart.shape[0], 1, 1))), axis=-2)

  ## Create motion distorted point cloud
  # number of state times
  num_meas_poses = len(phis)
  # setup velocity prior
  delt = 0.1
  velocity_prior = np.array([[-2.0, 0., 0., 0., 0., 0.4]]).T
  init_pose_vec = np.array([[0., 0., 0., 0., 0., 0.]]).T
  init_pose = Transformation(xi_ab=init_pose_vec)
  total_time = delt * (num_meas_poses - 1)
  # ground truth poses at measurements
  gt_T_k0 = [se3op.vec2tran(i * delt * velocity_prior) @ init_pose.matrix() for i in range(num_meas_poses)]
  # measurement times of each point
  ts = np.repeat(np.array([i * delt for i in range(num_meas_poses)]), len(thetas))
  # distorted point cloud
  qry_points = np.copy(ref_points)
  for i, T_k0 in enumerate(gt_T_k0):
    qry_points[i * len(thetas):(i + 1) * len(thetas)] = (T_k0 @ qry_points[i * len(thetas):(i + 1) * len(thetas)])
  # correspondences
  sample_inds = np.array([np.arange(qry_points.shape[0]), np.arange(qry_points.shape[0])]).T

  ## Setup states
  # initial conditions
  init_pose_vec = np.array([[0., 0., 0., 0., 0., 0.]]).T
  init_pose = Transformation(xi_ab=init_pose_vec)
  init_velocity = np.zeros((6, 1))
  # only need two pose and velocity states
  T_00_var = TransformStateVar(init_pose, is_locked=True, copy=True)
  T_00_eval = TransformStateEvaluator(T_00_var)
  w_0 = VectorSpaceStateVar(init_velocity, copy=True)
  T_10_var = TransformStateVar(init_pose, copy=True)
  T_10_eval = TransformStateEvaluator(T_10_var)
  w_1 = VectorSpaceStateVar(init_velocity, copy=True)

  ## Setup trajectory
  Qc_inv = np.diag(1 / np.array([1.0, 0.001, 0.001, 0.001, 0.001, 1.0]))  # smoothing factor diagonal
  traj = TrajectoryInterface(Qc_inv=Qc_inv)
  traj.add_knot(time=Time(secs=0), T_k0=T_00_eval, velocity=w_0)
  traj.add_knot(time=Time(secs=total_time), T_k0=T_10_eval, velocity=w_1)

  ## Setup shared noise and loss function
  noise_model = StaticNoiseModel(np.eye(3))
  loss_func = L2LossFunc()

  ## Setup cost terms
  cost_terms = []
  for i in range(qry_points.shape[0]):
    qry_pt = qry_points[sample_inds[i, 0]]
    ref_pt = ref_points[sample_inds[i, 1]]
    T_mq_eval = InverseTransformEvaluator(traj.get_interp_pose_eval(Time(secs=ts[sample_inds[i, 0]])))
    error_func = PointToPointErrorEval(T_mq_eval, ref_pt, qry_pt)
    cost_terms.append(WeightedLeastSquareCostTerm(error_func, noise_model, loss_func))

  ## Make optimization problem
  opt_prob = OptimizationProblem()
  opt_prob.add_state_var(T_00_var, w_0, T_10_var, w_1)
  opt_prob.add_cost_term(*cost_terms)

  ## Make solver and solve
  optimizer = GaussNewtonSolver(opt_prob, verbose=True)
  optimizer.optimize()

  print("Estimated Pose:     \n", T_10_var.get_value())
  print("Ground Truth Pose:  \n", gt_T_k0[-1])
  assert np.allclose(T_10_var.get_value().matrix(), gt_T_k0[-1])

  ## Get interpolated poses to align point cloud
  aligned = np.empty_like(ref_points)
  for i in range(ref_points.shape[0]):
    idx = sample_inds[i, 0]
    T_mq = traj.get_interp_pose_eval(Time(secs=ts[idx])).evaluate().inverse()
    aligned[idx] = T_mq.matrix() @ qry_points[idx]

  ## Plot if figure is given
  if fig is not None:
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Point Cloud Alignment with Motion Correction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(ref_points[:, 0, 0], ref_points[:, 1, 0], ref_points[:, 2, 0], s=10, c='g', label='Reference Points')
    ax.scatter(qry_points[:, 0, 0],
               qry_points[:, 1, 0],
               qry_points[:, 2, 0],
               s=10,
               c='r',
               label='Query Points (Motion Distorted)')
    ax.scatter(aligned[:, 0, 0], aligned[:, 1, 0], aligned[:, 2, 0], s=10, c='b', label='Aligned Points')
    ax.legend()


if __name__ == "__main__":
  without_motion_correction(plt.figure(0))
  with_motion_correction(plt.figure(1))
  plt.show()