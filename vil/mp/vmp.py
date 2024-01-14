import logging
from typing import Union
import numpy as np
import robot_utils
from .mp_common import can_sys, TrajectoryMP, pinv_rcond


class VMP(TrajectoryMP):
    def __init__(
            self,
            dim:                        int,
            kernel_num:                 int = 30,
            kernel_std:                 float = 0.1,
            elementary_type:            str = 'linear',
            use_out_of_range_kernel:    bool = True
    ):
        super(VMP, self).__init__(dim, kernel_num, kernel_std, elementary_type, use_out_of_range_kernel)

    def train(self, trajectories: np.ndarray, estimate_derivative: bool = False):
        """
        Assume trajectories are regularly sampled time-sequences. (N_demo, T, D)
        """
        if trajectories.ndim == 2:
            trajectories = np.expand_dims(trajectories, 0)

        n_demo, self.n_samples, dim = trajectories.shape
        if self.dim != dim - 1:
            logging.warning(f"the dimension of the trajectory {dim-1} doesn't match the MP's dimension {self.dim}. "
                            f"Updating MP's dimension to match the trajectory")
        self.dim = dim - 1

        linear_dim = self.dim + 1
        self._set_start_goal_for_training(trajectories, estimate_derivative, linear_dim)

        can_value_array = can_sys(1, 0, self.n_samples)
        psi = self.__psi__(can_value_array)                                                 # (T, K)
        linear_traj = self.linear_traj(can_value_array)                                     # (T, dim)
        shape_traj = trajectories[..., 1:] - np.expand_dims(linear_traj, 0)  # (N, T, dim) - (1, T, dim)

        pseudo_inv = np.linalg.pinv(psi.T.dot(psi), pinv_rcond)                             # (K, K)
        weights_over_demos = np.einsum("ij,njd->nid", pseudo_inv.dot(psi.T), shape_traj)  # (N, K, dim)
        self.kernel_weights = weights_over_demos.mean(axis=0)  # empirical mean (K, dim)
        self.empirical_variance = ((weights_over_demos - self.kernel_weights) ** 2).mean(axis=0)

        # shape_traj = trajectories[..., 1:].mean(0) - linear_traj                            # (T, dim) - (T, dim)
        # pseudo_inv = np.linalg.pinv(psi.T.dot(psi), pinv_rcond)                             # (K, K)
        # self.kernel_weights = np.einsum("ij,jd->id", pseudo_inv.dot(psi.T), shape_traj)     # (K, dim)

    def set_start_goal(
            self,
            start:  Union[np.ndarray, None] = None,
            goal:   Union[np.ndarray, None] = None,
            dy0:    Union[np.ndarray, None] = None,
            ddy0:   Union[np.ndarray, None] = None,
            dg:     Union[np.ndarray, None] = None,
            ddg:    Union[np.ndarray, None] = None
    ):
        self._set_start_goal(start, goal, None, dy0, ddy0, dg, ddg)

    def get_target(self, can_value: Union[float, np.ndarray]):
        return self.linear_traj(can_value) + self.shape_modulation(can_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vmp = VMP(1, kernel_num=10, use_out_of_range_kernel=True, elementary_type="linear")
    # vmp = VMP(1, kernel_num=10, use_out_of_range_kernel=True, elementary_type="min_jerk")
    t = np.linspace(0, 1, 1000)
    # noise_scale = 0.0
    noise_scale = 0.2
    traj0 = np.stack([t, np.sin(t * 2 * np.pi) + t + noise_scale * (np.random.random(t.shape) - 0.5)])
    traj0 = np.transpose(traj0)
    # traj1 = np.stack([t, np.cos(t * 2 * np.pi)])
    traj1 = np.stack([t, np.sin(t * 2 * np.pi) + t + noise_scale * (np.random.random(t.shape) - 0.5)])
    traj1 = np.transpose(traj1)

    trajs = np.stack([traj0, traj1])
    ic(trajs.shape)
    vmp.train(trajs)
    g = np.array([1.0])
    Xi = vmp.roll(start=np.array([0]), goal=g) #y0=(traj0[0,1:]+traj1[0,1:])/2, g=(traj0[-1,1:]+traj1[-1,1:])/2)
    # ic(vmp.centers.flatten())
    # ic(vmp.kernel_weights.flatten())
    lin_traj = vmp.linear_traj(np.linspace(1, 0, vmp.n_samples))
    xi = vmp.get_target(1-0.75)
    ic(xi.shape)

    plt.plot(t, traj0[:, 1:], 'r', linewidth=10, alpha=0.3)
    plt.plot(t, traj1[:, 1:], 'g', linewidth=5, alpha=0.3)
    plt.plot(t, Xi[:, 1:], 'b')
    plt.plot(t, lin_traj, "orange", linestyle="--")
    plt.scatter(0.75, xi, s=1000, color="yellow")
    plt.show()

