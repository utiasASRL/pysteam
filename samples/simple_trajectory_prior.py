"""Example that uses a simple constant-velocity prior over a trajectory."""
import numpy as np

from pylgmath import Transformation
from pysteam.trajectory import Time, TrajectoryInterface
from pysteam.state import TransformStateVar, VectorSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.solver import DoglegGaussNewtonSolver
from pysteam.evaluator import TransformStateEvaluator

np.set_printoptions(linewidth=160, precision=4)


class TrajStateVar:

  def __init__(
      self,
      time: Time,
      pose: TransformStateVar,
      velocity: VectorSpaceStateVar,
  ) -> None:
    self.time: Time = time
    self.pose: TransformStateVar = pose
    self.velocity: VectorSpaceStateVar = velocity


def main():
  ## Setup Problem
  num_poses = 100  # number of state times
  # setup velocity prior
  v_x = -1.0
  omega_z = 0.01
  velocity_prior = np.array([[v_x, 0., 0., 0., 0., omega_z]]).T
  # calculate time between states
  total_time = 2.0 * np.pi / omega_z  # time to do one circle
  delt = total_time / (num_poses - 1)
  # smoothing factor diagonal
  Qc_inv = np.diag(1 / np.array([1.0, 0.001, 0.001, 0.001, 0.001, 1.0]))

  ## Setup initial conditions
  init_pose_vec = np.array([[1., 2., 3., 4., 5., 6.]]).T
  init_pose = Transformation(xi_ab=init_pose_vec)
  init_velocity = np.zeros((6, 1))

  ## Setup states
  # steam state variables
  states = [
      TrajStateVar(
          Time(secs=i * delt),
          TransformStateVar(init_pose, copy=True),
          VectorSpaceStateVar(init_velocity, copy=True),
      ) for i in range(num_poses)
  ]
  # trajectory smoothing
  traj = TrajectoryInterface(Qc_inv=Qc_inv)
  for state in states:
    traj.add_knot(time=state.time, T_k0=TransformStateEvaluator(state.pose), velocity=state.velocity)

  ## Setup cost terms
  traj.add_pose_prior(Time(), init_pose, np.eye(6))
  traj.add_velocity_prior(Time(), velocity_prior, np.eye(6))

  ## Setup optimization problem
  opt_prob = OptimizationProblem()
  opt_prob.add_cost_term(*traj.get_prior_cost_terms())
  opt_prob.add_state_var(*[j for i in states for j in (i.pose, i.velocity)])

  ## Make the solver and solve
  optimizer = DoglegGaussNewtonSolver(opt_prob, verbose=True)
  optimizer.optimize()

  ## Get velocity at interpolated time
  curr_vel = traj.get_interp_velocity(states[0].time + Time(secs=0.5 * delt))

  ## Print results
  print("First Pose:                 \n", states[0].pose.get_value())
  print("Second Pose:                \n", states[1].pose.get_value())
  print("Last Pose (full circle):    \n", states[-1].pose.get_value())
  print("First Vel:                  \n", states[0].velocity.get_value().T)
  print("Second Vel:                 \n", states[1].velocity.get_value().T)
  print("Last Vel:                   \n", states[-1].velocity.get_value().T)
  print("Interp. Vel (t=t0+0.5*delT):\n", curr_vel.T)


if __name__ == "__main__":
  main()
