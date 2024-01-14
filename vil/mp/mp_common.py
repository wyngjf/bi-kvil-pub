import numpy as np
from scipy.signal import savgol_filter
from typing import Union

pinv_rcond = 1.4e-08


def can_sys(t0: float = 1, t1: float = 0, n_sample: int = 20):
    """
    return the sampled values of linear decay canonical system

    Args:
        t0: start time point
        t1: end time point
        n_sample: number of samples
    """
    return np.linspace(t0, t1, n_sample)


def get_smooth_traj(trajectory: np.ndarray, order: int = 0):
    delta = default_time_duration / pos_traj.shape[-1]
    vel = []
    for n in range(pos_traj.shape[0]):
        vel.append(savgol_filter(pos_traj[n], window_length=window_length, polyorder=polyorder, deriv=1, axis=-1,
                                 mode="nearest", delta=delta))
    vel = np.vstack(vel).reshape(pos_traj.shape)


def normalize_timestamp(trajectory: np.ndarray):
    """
    we assume that all the demonstrated trajectories evolve with the same peace. So only the first
    trajectory is taken as reference.

    Args:
        trajectory: (N, T, dim)

    Returns: the trajectories with normalized timestamps

    """
    trajectory[:, 0] = (trajectory[:, 0] - trajectory[0, 0])/(trajectory[-1, 0] - trajectory[0, 0])
    return trajectory


def get_min_jerk_params(y0, dy0, ddy0, g, dg, ddg):
    b = np.stack([y0, dy0, ddy0, g, dg, ddg])
    coeff_matrix = np.array(
        [[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20],
         [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
    )
    return np.linalg.solve(coeff_matrix, b)


class TrajectoryMP:
    def __init__(
            self,
            dim: int,
            kernel_num: int = 30,
            kernel_std: float = 0.1,
            elementary_type: str = 'linear',
            use_out_of_range_kernel: bool = True
    ):
        if dim <= 0:
            raise ValueError(f"dim should be larger than zero, give {dim}")
        if kernel_num <= 1:
            raise ValueError(f"the number of kernels should be larger than zero, give {kernel_num}")
        self.kernel_num = kernel_num
        if use_out_of_range_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)  # (K, )
        else:
            self.centers = np.linspace(1, 0, kernel_num)  # (K, )

        self.kernel_variance = kernel_std ** 2
        self.var_reci = - 0.5 / self.kernel_variance
        self.elementary_type = elementary_type
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.kernel_weights = np.zeros(shape=(kernel_num, self.dim))
        self.empirical_variance = np.zeros(shape=(kernel_num, self.dim))

        self.h_params = None
        self.start = None
        self.goal = None

    def __psi__(self, can_value: Union[float, np.ndarray]) -> np.ndarray:
        """
        compute the value of each kernel given canonical values

        Args:
            can_value: float or numpy array of size (T, ) or (T, 1)

        Returns: the value (array) of the kernels at can_value of size (K, ) or (T, K)

        """
        if np.ndim(can_value) == 0:
            can_value = np.array([[can_value]])
        elif np.ndim(can_value) == 1:
            can_value = can_value[..., np.newaxis]
        return np.exp(np.square(can_value - self.centers) * self.var_reci).squeeze()

    def __dpsi__(self, can_value: Union[float, np.ndarray]) -> np.ndarray:
        fx = self.__psi__(can_value)
        dft = - np.multiply(fx, 2 * (can_value - self.centers) * self.var_reci)
        return dft

    def shape_modulation(self, can_values: Union[float, np.ndarray]) -> np.ndarray:
        """
        return the shape trajectory

        Args:
            can_values: float, or np array (T, ) or (T, 1). if float: psi (K, ) * weights (K, dim) -> (dim, );
                if array, psi (T, K) * weights (K, dim) -> (T, dim)

        Returns: shape trajectory of size (dim, ) or (T, dim)

        """
        res = np.matmul(self.__psi__(can_values), self.kernel_weights)
        return res

    def linear_traj(self, can_values: Union[float, np.ndarray]) -> np.ndarray:
        """
        compute the linear trajectory

        Args:
            can_values: float, or numpy array of size (T, ) or (T, 1)

        Returns: linear trajectory (dim, ) or (T, dim)

        """
        if np.ndim(can_values) == 0:
            can_values = np.array([can_values])
        elif np.ndim(can_values) == 2:
            can_values = can_values.squeeze()

        if self.elementary_type == 'linear':
            can_values_aug = np.stack([np.ones_like(can_values), can_values])   # (2, T)
        else:  # min_jerk
            can_values_aug = np.stack([     # (6, T)
                np.ones_like(can_values),
                can_values,
                np.power(can_values, 2),
                np.power(can_values, 3),
                np.power(can_values, 4),
                np.power(can_values, 5)
            ])
        # linear:   (2, dim) (2, T) -> (T, dim)
        # min_jerk: (6, dim) (6, T) -> (T, dim)
        traj = np.einsum("ij,ik->kj", self.h_params, can_values_aug)
        if traj.shape[0] == 1:
            return traj.squeeze(0)
        else:
            return traj

    @staticmethod
    def get_average_point(points: np.ndarray, axis: int = 0):
        """

        Args:
            points: (T, dim) or (N, T, dim)
            axis: default 0, where the points should have dimension (n_demo, time, dim)

        Returns: (dim, ) or (T, dim)

        """
        return np.mean(points, axis=axis)

    def save_weights_to_file(self, filename: str):
        np.savetxt(filename, self.kernel_weights, delimiter=',')

    def load_weights_from_file(self, filename: str):
        self.kernel_weights = np.loadtxt(filename, delimiter=',')

    def get_weights(self) -> np.ndarray:
        return self.kernel_weights

    def get_flatten_weights(self):
        return self.kernel_weights.flatten('F')

    def set_weights(self, weights: np.ndarray):
        """
        set weights to VMP

        Args:
            weights: (kernel_num, dim)
        """
        if np.shape(weights)[-1] == self.dim * self.kernel_num:
            self.kernel_weights = np.reshape(weights, (self.kernel_num, self.dim), 'F')
        elif np.shape(weights)[0] == self.kernel_num and np.shape(weights)[-1] == self.dim:
            self.kernel_weights = weights
        else:
            raise Exception(f"The weights have wrong shape. "
                            f"It should have {self.kernel_num} rows (for kernel number) "
                            f"and {self.dim} columns (for dimensions), but given is {weights.shape}.")

    def _check_start_goal(
            self,
            start: Union[np.ndarray, None] = None,
            goal: Union[np.ndarray, None] = None,
    ):
        if start is None:
            start = self.start
        if goal is None:
            goal = self.goal
        if np.ndim(start) == 2:
            start = start.flatten()
        if np.ndim(goal) == 2:
            goal = goal.flatten()
        return start, goal

    def _set_start_goal(
            self,
            start:  Union[np.ndarray, None] = None,
            goal:   Union[np.ndarray, None] = None,
            dim:    Union[int, None] = None,
            dy0:    Union[np.ndarray, None] = None,
            ddy0:   Union[np.ndarray, None] = None,
            dg:     Union[np.ndarray, None] = None,
            ddg:    Union[np.ndarray, None] = None
    ):
        start, goal = self._check_start_goal(start, goal)

        if self.elementary_type == "min_jerk":
            if dim is None:
                dim = self.dim

            def valid(x):
                return x is not None and np.shape(x)[0] == dim

            if not valid(dy0):
                dy0 = np.zeros_like(start)
            if not valid(ddy0):
                ddy0 = np.zeros_like(start)
            if not valid(dg):
                dg = np.zeros_like(start)
            if not valid(ddg):
                ddg = np.zeros_like(start)

            self.h_params = get_min_jerk_params(start, dy0, ddy0, goal, dg, ddg)
        else:
            self.h_params = np.stack([goal, start - goal])

    def set_start_goal(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            dy0: Union[np.ndarray, None] = None,
            ddy0: Union[np.ndarray, None] = None,
            dg: Union[np.ndarray, None] = None,
            ddg: Union[np.ndarray, None] = None
    ):
        raise NotImplementedError

    def get_target(self, can_value: Union[float, np.ndarray]):
        raise NotImplementedError

    def _set_start_goal_for_training(
            self,
            trajectories: np.ndarray,
            estimate_derivative: bool = False,
            linear_dim: int = 0
    ):
        if self.elementary_type == "linear":
            start = self.get_average_point(trajectories[:, 0])[1:]   # (dim, )
            goal = self.get_average_point(trajectories[:, -1])[1:]   # (dim, )
            dy0, ddy0, dg, ddg = None, None, None, None

        else:  # min_jerk
            start = self.get_average_point(trajectories[:, 0:3])    # (3, dim+1)
            goal = self.get_average_point(trajectories[:, -3:])     # (3, dim+1)
            if estimate_derivative:
                dy0 = -(start[1, 1:linear_dim] - start[0, 1:linear_dim]) / (start[1, 0] - start[0, 0])  # (dim, )
                dy1 = -(start[2, 1:linear_dim] - start[1, 1:linear_dim]) / (start[2, 0] - start[1, 0])  # (dim, )
                ddy0 = -(dy1 - dy0) / (start[1, 0] - start[0, 0])                                       # (dim, )
                dg0 = -(goal[1, 1:linear_dim] - goal[0, 1:linear_dim]) / (goal[1, 0] - goal[0, 0])      # (dim, )
                dg = -(goal[2, 1:linear_dim] - goal[1, 1:linear_dim]) / (goal[2, 0] - goal[1, 0])       # (dim, )
                ddg = -(dg - dg0) / (goal[1, 0] - goal[0, 0])                                           # (dim, )
            else:
                dy0, ddy0, dg, ddg = None, None, None, None

            start = start[0, 1:]
            goal = goal[-1, 1:]

        self.set_start_goal(start, goal, dy0, ddy0, dg, ddg)
        self.start = start
        self.goal = goal

    def roll(
            self,
            start:      np.ndarray,
            goal:       np.ndarray,
            n_samples:  Union[int, None] = None
    ) -> np.ndarray:
        """
        reproduce the trajectory

        Args:
            start: (dim, ) or (dim, 1) start point y0
            goal: (dim, ) or (dim, 1) end point g
            n_samples: number of samples

        Returns: reproduced trajectory (n_samples, dim)

        """
        n_samples = self.n_samples if n_samples is None else n_samples
        can_values = can_sys(1, 0, n_samples)
        self.set_start_goal(start, goal)
        traj = self.get_target(can_values)

        time_stamp = 1 - can_values[..., np.newaxis]
        return np.concatenate([time_stamp, traj], axis=1)



