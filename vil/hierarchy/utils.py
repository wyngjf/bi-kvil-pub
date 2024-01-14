# import torch
# import math
import numpy as np
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
_EPS = np.finfo(float).eps * 4.0


# def quaternion_conjugate(quaternion):
#     """Return conjugate of quaternion.
#
#     >>> q0 = random_quaternion()
#     >>> q1 = quaternion_conjugate(q0)
#     >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
#     True
#
#     """
#     q = np.array(quaternion, dtype=np.float64, copy=True)
#     np.negative(q[1:], q[1:])
#     return q


# def quaternion_multiply(quaternion1, quaternion0):
#     """Return multiplication of two quaternions.
#
#     >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
#     >>> np.allclose(q, [28, -44, -14, 48])
#     True
#
#     """
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return np.array([
#         -x1*x0 - y1*y0 - z1*z0 + w1*w0,
#         x1*w0 + y1*z0 - z1*y0 + w1*x0,
#         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
#         x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


# def quaternion_rotate(ori_quat, rot_quat):
#     return quaternion_multiply(quaternion_multiply(rot_quat, ori_quat), quaternion_conjugate(rot_quat))


# def quaternion_average(quat, weights=None, keepdim=False):
#     """
#     compute the weighted average quaternion. see the comments of this answer https://stackoverflow.com/a/49690919
#
#     Args:
#         quat: (batch, 4, ...)
#         weights: weight coefficients
#
#     Returns: (..., 4), e.g. input (batch, 4) --> (4, ), input (batch, 4, time) --> (time, 4)
#
#     """
#     if len(quat.shape) < 2:
#         raise ValueError("quat should have shape (batch, 4, ...)")
#     if weights is None:
#         result = torch.linalg.eigh(torch.einsum('ij...,ik...->...jk', quat, quat))[1][:, -1]
#     else:
#         if quat.shape[0] != len(weights):
#             raise ValueError("number of quaternions and weights mismatch!")
#         result = torch.linalg.eigh(torch.einsum('ij...,ik...,i->...jk', quat, quat, weights))[1][:, -1]
#     if len(result.size()) == 2:
#         result = result.transpose(1, 0)
#     if keepdim:
#         return result.unsqueeze(0)
#     else:
#         return result

#
# def get_mesh_grid(side_length, dim=2, range=None, reshape=True):
#     """
#     Generates a flattened grid of (x,y,...) coordinates
#     Args:
#         side_length: int or list/tuple of ints. int, generate same number of samples for each dim.
#                      list, generate different number of samples for each dim
#         dim: when side_length is int, you need to specify dimension of the coordinates
#         range: a list of tuple, [(min, max), (min, max) ... ] specify the sample range of each dim
#
#     Returns: flattened grid as 2D matrix, each row is a sampled coordinates
#
#     """
#     # tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length, dtype=torch.float64)])
#     if isinstance(side_length, int):
#         if range is None:
#             tensors = tuple(dim * [torch.linspace(-1, 1, steps=side_length)])
#         else:
#             tensors = tuple(dim * [torch.linspace(range[0], range[1], steps=side_length)])
#     else:
#         if range is None:
#             tensors = tuple([torch.linspace(-1, 1, steps=s) for s in side_length])
#         else:
#             tensors = tuple([torch.linspace(r[0], r[1], steps=s) for s, r in zip(side_length, range)])
#     mesh_grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
#     if reshape:
#         mesh_grid = mesh_grid.reshape(-1, dim)
#     return mesh_grid


# def get_covariance(time_steps=30, l=2, sig_f=1, sig_n=0.1):
#     mesh = get_mesh_grid(time_steps, dim=2, range=[1, time_steps])
#     K = np.square(sig_f) * np.exp(- np.square(mesh[:, 0] - mesh[:, 1]) / (2 * np.square(l)))
#     K = K.reshape(time_steps, time_steps) + sig_n * torch.eye(time_steps)
#     tri = torch.linalg.cholesky(K)
#     return torch.einsum("ij,kj->ik", tri, tri)


def sphere_logarithmic_map(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    distance = np.arccos(np.clip(np.dot(x0.T, x), -1., 1.))
    denominator = np.sin(distance)
    denominator[denominator < 1e-16] += 1e-16
    u = (x - x0 * np.cos(distance)) * distance/denominator

    u[:, distance[0] < 1e-16] = np.zeros((u.shape[0], 1))
    return u


def sphere_exponential_map(u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    This function maps a vector u lying on the tangent space of x0 into the manifold.

    Parameters
    ----------
    :param u: vector in the tangent space
    :param x0: basis point of the tangent space

    Returns
    -------
    :return: x: point on the manifold
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(u) < 2:
        u = u[:, None]

    norm_u = np.sqrt(np.sum(u*u, axis=0))
    x = x0 * np.cos(norm_u) + u * np.sin(norm_u)/norm_u

    x[:, norm_u < 1e-16] = x0

    return x


# def multivariate_normal(x, mu, sigma=None, log=True):
#     """
#     Multivariate normal distribution
#
#     :param x:   np.array([nb_samples, nb_dim])
#     :param mu:    np.array([nb_dim])
#     :param sigma:   np.array([nb_dim, nb_dim])
#     :param log:   bool
#     :return:
#     """
#     dx = x - mu
#     if sigma.ndim == 1:
#         sigma = sigma[:, None]
#         dx = dx[:, None]
#         inv_sigma = np.linalg.inv(sigma)
#         log_lik = -0.5 * np.sum(np.dot(dx, inv_sigma) * dx, axis=1) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))
#     else:
#         inv_sigma = np.linalg.inv(sigma)
#         log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', inv_sigma, dx)) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))
#
#     return log_lik if log else np.exp(log_lik)


# def set_3d_equal_auto(ax, xlim=None, ylim=None, zlim=None):
#     ax.set_box_aspect((1, 1, 1))
#     if xlim is None:
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         zlim = ax.get_zlim()
#     d_x = xlim[1] - xlim[0]
#     d_y = ylim[1] - ylim[0]
#     d_z = zlim[1] - zlim[0]
#     max_d = 0.5 * max([d_x, d_y, d_z])
#     mean_x = 0.5 * (xlim[0] + xlim[1])
#     mean_y = 0.5 * (ylim[0] + ylim[1])
#     mean_z = 0.5 * (zlim[0] + zlim[1])
#     ax.set_xlim(mean_x - max_d, mean_x + max_d)
#     ax.set_ylim(mean_y - max_d, mean_y + max_d)
#     ax.set_zlim(mean_z - max_d, mean_z + max_d)


# def set_3d_ax_label(ax, labels: list):
#     ax.set_xlabel(labels[0])
#     ax.set_ylabel(labels[1])
#     ax.set_zlabel(labels[2])


# def quaternion_matrix(quaternion, rot_only=False):
#     """
#     Return homogeneous rotation matrix from quaternion. (w, x, y, z)
#     """
#     q = np.array(quaternion, dtype=np.float64, copy=True)
#     n = np.dot(q, q)
#     if n < _EPS:
#         return np.identity(4)
#     q *= math.sqrt(2.0 / n)
#     q = np.outer(q, q)
#     mat = np.array([
#         [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
#         [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
#         [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
#         [                0.0,                 0.0,                 0.0, 1.0]])
#     if rot_only:
#         return mat[:3, :3]
#     return mat

#
# def draw_frame_3d(ax, config: np.ndarray, scale: float = 1.0, alpha: float = 0.5, collections=None):
#     """
#     Args:
#         ax: matplotlib axes to plot such frame
#         config: configuration of the local frame, it can be multiple format, e.g.
#             (1) position + quaternion [x, y, z, qw, qx, qy, qz], 7 DoF
#             (2) position + flattened 3x3 rotation matrix, 3 + 9 = 12 DoF
#             (3) position + RPY Euler angle, 6 DoF
#         scale: the scaling of the arrow, default is 1.0
#         alpha:
#     """
#     arrow = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
#
#     if config.shape[0] == 4:
#         config = np.concatenate((np.zeros(3), config))
#
#     if config.shape[0] == 7:
#         rot_mat = quaternion_matrix(config[3:], rot_only=True)
#     else:
#         raise NotImplementedError
#     arrow = np.einsum("ij, kj->ki", rot_mat, arrow)
#     arrow_line_collect = np.concatenate((np.tile(config[:3], (3, 1)), arrow + config[:3]), axis=-1).reshape((3, 2, 3)).tolist()
#
#     if collections is None:
#         collections = ax.add_collection3d(Line3DCollection(arrow_line_collect, color=['orangered', 'g', 'b'], linewidths=5, alpha=alpha))
#     else:
#         collections.set_segments(arrow_line_collect)
#
#     return collections