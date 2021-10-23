"""Simple nonlinear 3D estimation problem using the Starry Night Dataset. See AER1513 assignment 3 for a description.
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import matplotlib.pyplot as plt
from scipy.io import loadmat

from pylgmath import so3op, se3op, Transformation
from pysteam import state, evaluator, problem, solver


class Estimator:

  def __init__(self, dataset):
    # load data
    data = loadmat(dataset)

    # total time steps
    self.K = data["t"].shape[-1]

    # stereo camera
    self.f_u = data["fu"][0, 0]
    self.f_v = data["fv"][0, 0]
    self.c_u = data["cu"][0, 0]
    self.c_v = data["cv"][0, 0]
    self.b = data["b"][0, 0]

    # stereo camera and imu
    C_cv, rho_cv_inv = data["C_c_v"], data["rho_v_c_v"]
    self.T_cv = se3op.Cr2T(C_ab=C_cv, r_ba_ina=-C_cv @ rho_cv_inv)

    # ground truth values
    r_vi_ini = data["r_i_vk_i"].T[..., None]
    C_vi = so3op.vec2rot(data["theta_vk_i"].T[..., None]).swapaxes(-1, -2)
    self.T_vi = se3op.Cr2T(C_ab=C_vi, r_ba_ina=-C_vi @ r_vi_ini)  # this is the ground truth

    # inputs
    w_vi_inv, v_vi_inv = data["w_vk_vk_i"].T, data["v_vk_vk_i"].T
    self.varpi_iv_inv = np.concatenate([-v_vi_inv, -w_vi_inv], axis=-1)[..., None]
    self.t = data["t"].squeeze()  # time steps (1900,)
    ts = np.roll(self.t, 1)
    ts[0] = 0
    self.dt = self.t - ts

    # measurements
    rho_pi_ini = data["rho_i_pj_i"].T[..., None]  # feature positions (20 x 3 x 1)
    rho_pi_ini = np.repeat(rho_pi_ini[None, ...], self.K, axis=0)  # feature positions (1900, 20 x 3 x 1)
    padding = np.ones(rho_pi_ini.shape[:-2] + (1,) + rho_pi_ini.shape[-1:])
    self.rho_pi_ini = np.concatenate((rho_pi_ini, padding), axis=-2)  # feature positions (1900, 20 x 4 x 1)
    self.y_k_j = data["y_k_j"].transpose((1, 2, 0))[..., None]  # measurements (1900, 20, 4, 1)
    self.y_filter = np.where(self.y_k_j == -1, 0, 1)  # [..., 0, 0] # filter (1900, 20, 4, 1)

    # covariances
    w_var, v_var, y_var = data["w_var"], data["v_var"], data["y_var"]
    w_var_inv = np.reciprocal(w_var.squeeze())
    v_var_inv = np.reciprocal(v_var.squeeze())
    y_var_inv = np.reciprocal(y_var.squeeze())
    self.Q_inv = np.zeros((self.K, 6, 6))
    self.Q_inv[..., :, :] = spla.block_diag(np.diag(v_var_inv), np.diag(w_var_inv))
    self.R_inv = np.zeros((*(self.y_k_j.shape[:2]), 4, 4))
    self.R_inv[..., :, :] = np.diag(y_var_inv)

    # helper matrices
    self.D = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # estimated values of variables
    self.hat_T_vi = np.zeros_like(self.T_vi)
    self.hat_T_vi[...] = self.T_vi[...]  # estimate of poses initialized to be ground truth
    self.hat_P = np.zeros((self.K, 6, 6))
    self.hat_P[..., :, :] = np.eye(6) * 1e-4
    self.hat_stds = np.ones((self.K, 6)) * np.sqrt(1e-4)

    self.k1 = 0
    self.k2 = self.K

  def set_interval(self, k1=None, k2=None):
    self.k1 = k1 if k1 != None else self.k1
    self.k2 = k2 if k2 != None else self.k2

  def initialize(self, k1=None, k2=None):
    """
    Initialize a portion of the states between k1 and k2 using dead reckoning, starting with the current estimate of k1
    """
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    for k in range(k1 + 1, k2):
      self.hat_T_vi[k] = self.f(self.hat_T_vi[k - 1], self.varpi_iv_inv[k - 1], self.dt[k])
      # covariance
      F = self.df(self.hat_T_vi[k - 1], self.varpi_iv_inv[k - 1], self.dt[k])
      Q_inv = self.Q_inv[k]
      Q_inv = Q_inv / (self.dt[k, None, None]**2)
      self.hat_P[k] = F @ self.hat_P[k - 1] @ F.T + npla.inv(Q_inv)

  def optimize(self, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    # setup state variables with initial condition
    T_k0_vars = [state.TransformStateVar(Transformation(T_ba=self.hat_T_vi[i])) for i in range(k1, k2)]

    # setup loss function
    loss_func = problem.L2LossFunc()

    cost_terms = []
    # construct input cost terms
    for k in range(k1, k2 - 1):
      meas = Transformation(T_ba=se3op.vec2tran(self.varpi_iv_inv[k] * self.dt[k + 1]))
      Q_inv = self.Q_inv[k + 1] / (self.dt[k + 1]**2)
      Q_noise_model = problem.StaticNoiseModel(Q_inv, "information")
      error_func = evaluator.TransformErrorEval(meas_T_21=meas, T_10=T_k0_vars[k - k1], T_20=T_k0_vars[k - k1 + 1])
      cost_terms.append(problem.WeightedLeastSquareCostTerm(error_func, Q_noise_model, loss_func))
    # construct measurement cost terms
    intrinsics = evaluator.CameraIntrinsics(self.f_u, self.f_v, self.c_u, self.c_v, self.b)
    T_cv_var = evaluator.FixedTransformEvaluator(Transformation(T_ba=self.T_cv))
    R_noise_model = problem.StaticNoiseModel(self.R_inv[0, 0], "information")
    for k in range(k1, k2):
      for l in range(20):  # 20 is number of landmarks
        if self.y_filter[k, l, 0, 0] == 0:
          continue
        landmark = state.LandmarkStateVar(self.rho_pi_ini[k, l], is_locked=True)
        meas = self.y_k_j[k, l]
        T_k0 = evaluator.TransformStateEvaluator(T_k0_vars[k - k1])
        T_c0 = evaluator.ComposeTransformEvaluator(T_cv_var, T_k0)
        error_func = evaluator.StereoCameraErrorEval(meas, intrinsics, T_c0, landmark)
        cost_terms.append(problem.WeightedLeastSquareCostTerm(error_func, R_noise_model, loss_func))

    # construct the optimization problem
    opt_prob = problem.OptimizationProblem()
    opt_prob.add_cost_term(*cost_terms)
    opt_prob.add_state_var(*T_k0_vars)

    # construct the solver and solve
    optimizer = solver.GaussNewtonSolver(opt_prob, verbose=True, use_sparse_matrix=True)
    optimizer.optimize()

    # copy variables back
    for i, j in enumerate(range(k1, k2)):
      self.hat_T_vi[j] = T_k0_vars[i].get_value().matrix()

    # covariance
    full_hat_P = optimizer.query_covariance()
    self.hat_P[k1:k2] = np.array(
        [full_hat_P[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] for i in range(int(full_hat_P.shape[0] / 6))])
    # for plotting
    self.hat_stds[k1:k2] = (np.sqrt(np.diag(full_hat_P))).reshape((-1, 6))

  def plot_trajectory(self, fig, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    C_vi, r_iv_inv = se3op.T2Cr(T_ab=self.T_vi)
    r_vi_ini = -C_vi.swapaxes(-2, -1) @ r_iv_inv
    hat_C_vi, hat_r_iv_inv = se3op.T2Cr(T_ab=self.hat_T_vi)
    hat_r_vi_ini = -hat_C_vi.swapaxes(-2, -1) @ hat_r_iv_inv

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(hat_r_vi_ini[:, 0, 0], hat_r_vi_ini[:, 1, 0], hat_r_vi_ini[:, 2, 0], c='g', label='estimate')
    ax.plot(r_vi_ini[:, 0, 0], r_vi_ini[:, 1, 0], r_vi_ini[:, 2, 0], c='r', label='ground truth')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_xlim3d(0, 5)
    ax.set_ylim3d(0, 5)
    ax.set_zlim3d(0, 3)
    ax.set_title("Estimated Trajectory VS Ground Truth")
    ax.legend()

  def plot_error(self, fig, k1=None, k2=None):
    k1 = self.k1 if k1 is None else k1
    k2 = self.k2 if k2 is None else k2

    C_vi, r_iv_inv = se3op.T2Cr(T_ab=self.T_vi)
    r_vi_ini = -C_vi.swapaxes(-2, -1) @ r_iv_inv
    hat_C_vi, hat_r_iv_inv = se3op.T2Cr(T_ab=self.hat_T_vi)
    hat_r_vi_ini = -hat_C_vi.swapaxes(-2, -1) @ hat_r_iv_inv

    eye = np.zeros_like(C_vi)
    eye[..., :, :] = np.eye(3)
    rot_err = so3op.hatinv(eye - hat_C_vi @ npla.inv(C_vi))
    trans_err = hat_r_vi_ini - r_vi_ini

    t = self.t[k1:k2]
    stds = self.hat_stds[k1:k2, :]

    # plot landmarks for reference
    num_meas = np.sum(self.y_filter[k1:k2, :, 0, 0], axis=-1)
    green = np.argwhere(num_meas >= 3)
    red = np.argwhere(num_meas < 3)

    plot_number = 711
    fig.set_size_inches(8, 12)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.1, top=0.95, wspace=0.7, hspace=0.6)

    ax = fig.add_subplot(plot_number)
    ax.scatter(t[green], num_meas[green], s=1, c='g')
    ax.scatter(t[red], num_meas[red], s=1, c='r')
    ax.set_xlabel(r't [$s$]')
    ax.set_ylabel(r'Num. of Visible L.')

    labels = ['x', 'y', 'z']
    for i in range(3):
      ax = fig.add_subplot(plot_number + 1 + i)
      ax.plot(t, trans_err[k1:k2, i].flatten(), '-', linewidth=1.0)
      ax.plot(t, 3 * stds[:, i], 'r--', linewidth=1.0)
      ax.plot(t, -3 * stds[:, i], 'g--', linewidth=1.0)
      ax.fill_between(t, -3 * stds[:, i], 3 * stds[:, i], alpha=0.2)
      ax.set_xlabel(r"$t$ [$s$]")
      ax.set_ylabel(r"$\hat{r}_x - r_x$ [$m$]".replace("x", labels[i]))
    for i in range(3):
      ax = fig.add_subplot(plot_number + 4 + i)
      ax.plot(t, rot_err[k1:k2, i].flatten(), '-', linewidth=1.0)
      ax.plot(t, 3 * stds[:, 3 + i], 'r--', linewidth=1.0)
      ax.plot(t, -3 * stds[:, 3 + i], 'g--', linewidth=1.0)
      ax.fill_between(t, -3 * stds[:, 3 + i], 3 * stds[:, 3 + i], alpha=0.2)
      ax.set_xlabel(r"$t$ [$s$]")
      ax.set_ylabel(r"$\hat{\theta}_x - \theta_x$ [$rad$]".replace("x", labels[i]))

  def f(self, T, v, dt):
    """motion model"""
    dt = dt.reshape(-1, *([1] * len(v.shape[1:])))
    return se3op.vec2tran(dt * v) @ T

  def df(self, T, v, dt):
    """linearized motion model"""
    dt = dt.reshape(-1, *([1] * len(v.shape[1:])))
    return se3op.vec2jacinv(dt * v)


def main():
  estimator = Estimator("starry_night_dataset.mat")
  estimator.set_interval(1215, 1715)  # batch estimation between time stamp 1215 and 1715
  estimator.initialize()  # initialize with odometry
  estimator.optimize()  # estimator.optimize()
  estimator.plot_trajectory(plt.figure(0))
  estimator.plot_error(plt.figure(1))
  plt.show()


if __name__ == "__main__":
  main()