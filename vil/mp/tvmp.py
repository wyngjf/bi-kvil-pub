import numpy as np
from typing import Union

import robot_utils.math.transformations as tn
# from robot_utils.math.Quaternion import Quaternion

from .mp_common import can_sys, TrajectoryMP, get_min_jerk_params, pinv_rcond
from .mp_common import normalize_timestamp


class TVMP(TrajectoryMP):
    def __init__(
            self,
            kernel_num: int = 30,
            kernel_std: float = 0.1,
            elementary_type: str = 'linear',
            use_out_of_range_kernel: bool = True
    ):
        super(TVMP, self).__init__(7, kernel_num, kernel_std, elementary_type, use_out_of_range_kernel)

        self.q0, self.q1 = None, None

    @staticmethod
    def get_average_point(points: np.ndarray, axis: int = 0):
        """

        Args:
            points: (T, 8) or (N, T, 8), first dimension is time
            axis: default 0, where the points should have dimension (n_demo, time, dim+1)

        Returns: (8, ) or (T, 8)

        """
        points_translation, points_quaternion = np.split(points, [4, ], axis=-1)
        if np.ndim(points) == 3:
            quat_average = np.zeros((points.shape[1], 4))
            for i in range(points.shape[1]):
                quat_average[i] = tn.quaternion_average(points_quaternion[:, i])
        else:
            quat_average = tn.quaternion_average(points_quaternion)
        return np.concatenate((np.mean(points_translation, axis=axis), quat_average), axis=-1)

    def train(self, trajectories: np.ndarray, estimate_derivative: bool = False):
        """
        Assume trajectories are regularly sampled time-sequences. (N_demo, T, D)
        """
        if trajectories.ndim == 2:
            trajectories = np.expand_dims(trajectories, 0)

        if trajectories.shape[-1] != 8:
            raise ValueError("TVMP: can only be trained on 7 dimensional trajectories with one dimensional timestamp")

        n_demo, self.n_samples, _ = trajectories.shape

        linear_dim = 3 + 1
        self._set_start_goal_for_training(trajectories, estimate_derivative, linear_dim)

        can_value_array = can_sys(1, 0, self.n_samples)
        psi = self.__psi__(can_value_array)  # (T, K)
        linear_traj = self.linear_traj(can_value_array)  # (T, 3)
        shape_traj_translation = trajectories[..., 1:4] - linear_traj[np.newaxis, ...]  # (N, T, 3) - (1, T, 3)

        # calculate the quaternion trajectory
        q_start = tn.quaternion_average(trajectories[:, 0, 4:])
        q_end = tn.quaternion_average(trajectories[:, -1, 4:])
        q_geodesic = tn.quaternion_slerp(q_start, q_end, fraction=np.linspace(0, 1, self.n_samples))
        shape_traj_quat = tn.quaternion_rot_diff(q_geodesic, trajectories[..., 4:])

        shape_traj = np.concatenate([shape_traj_translation, shape_traj_quat], axis=-1)
        pseudo_inv = np.linalg.pinv(psi.T.dot(psi), pinv_rcond)  # (K, K)
        weights_over_demos = np.einsum("ij,njd->nid", pseudo_inv.dot(psi.T), shape_traj)  # (N, K, dim)
        self.kernel_weights = weights_over_demos.mean(axis=0)  # empirical mean (K, dim)
        self.empirical_variance = ((weights_over_demos - self.kernel_weights) ** 2).mean(axis=0)

    def get_target(self, can_value: Union[float, np.ndarray]):
        shape_traj = self.shape_modulation(can_value)
        traj_translation = self.linear_traj(can_value) + shape_traj[..., :3]

        q_geodesic = tn.quaternion_slerp(self.q0, self.q1, fraction=1 - can_value)
        traj_quaternion = tn.quaternion_multiply(q_geodesic, tn.quaternion_normalize(shape_traj[..., 3:]))
        return np.concatenate((traj_translation, traj_quaternion), axis=-1)

    def get_vel(self, t):
        raise NotImplementedError
        F = self.get_shape_modulation(t)
        dFt = self.get_shape_modulation(t, deri=1)
        translv = self.h_params[:, -1] + dFt[0:3]

        hq = Quaternion.slerp(t, self.q0, self.q1)
        fq = F[3:]

        dhq = Quaternion.slerp(t, self.q0, self.q1, deri=1)
        dfq = dFt[3:]
        rotv = Quaternion.qmulti(dhq, fq) + Quaternion.qmulti(hq, dfq)
        return np.expand_dims(translv, axis=0), np.expand_dims(rotv, axis=0)

    def set_start_goal(
            self,
            start:  Union[np.ndarray, None] = None,
            goal:   Union[np.ndarray, None] = None,
            dy0:    Union[np.ndarray, None] = None,
            dg:     Union[np.ndarray, None] = None,
            ddy0:   Union[np.ndarray, None] = None,
            ddg:    Union[np.ndarray, None] = None
    ):
        start, goal = self._check_start_goal(start, goal)
        self.q0, self.q1 = start[3:], goal[3:]
        self._set_start_goal(start[:3], goal[:3], 3, dy0, ddy0, dg, ddg)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tvmp = TVMP(elementary_type="min_jerk")
    # tvmp = TVMP(elementary_type="linear")
    trajs = np.loadtxt('./test/pickplace7d.csv', delimiter=',')
    trajs = normalize_timestamp(trajs)

    trajs = np.expand_dims(trajs, axis=0)
    tvmp.train(trajs)

    t = np.linspace(0, 1, 100)
    tvmp.set_start_goal(trajs[0, 0, 1:], trajs[0, -1, 1:])
    # ttraj = tvmp.roll(trajs[0, 0, 1:], trajs[0, -1, 1:])
    ttraj = []
    vtraj = []
    for i in range(100):
        pose = tvmp.get_target(1 - t[i])
        # translv, rotv = tvmp.get_vel(1 - t[i])

        ttraj.append(np.concatenate([np.array([t[i]]), pose], axis=-1))
        # vtraj.append(np.concatenate([np.array([[t[i]]]), translv, rotv], axis=1))

    ttraj = np.stack(ttraj)
    # vtraj = np.stack(vtraj)

    print('ttraj has shape: {}'.format(np.shape(ttraj)))
    for i in range(7):
        plt.figure(i)
        plt.plot(trajs[0, :, 0], trajs[0, :, i + 1], 'k-.')
        plt.plot(ttraj[:, 0], ttraj[:, i + 1], 'r-')

    plt.show()
