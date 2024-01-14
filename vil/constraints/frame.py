import copy
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from robot_utils.math.transformations import euler_matrix


def get_rot_mat_2d(alpha: float):
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def get_rot_mat_3d(euler: np.ndarray):
    return euler_matrix(*euler.tolist(), axes="sxyz")[:3, :3]


def get_loss(
        dim: int,
        origin: np.ndarray,
        orientation_config: np.ndarray,
        new_global_coords: np.ndarray,
        original_local_coords: np.ndarray
) -> float:
    """
    Compute the MSE loss given a frame configuration, current global coordinates and the expected local coordinates
    """
    # ic(orientation_config)
    # if dim == 3:
    #     rot_mat = quat2Mat(orientation_config)
    # else:
    #     rot_mat = get_rot_mat_2d(orientation_config[0])
    # # ic(new_global_coords.shape, origin.shape)
    # local_coords = np.einsum("ij, kj -> ki", rot_mat.T, (new_global_coords - origin))
    # # loss = np.square(np.linalg.norm((original_local_coords - local_coords), axis=-1)).mean()
    # loss = np.sum(np.abs(original_local_coords - local_coords), axis=-1).sum()
    # return loss
    if dim == 3:
        rot_mat = get_rot_mat_3d(orientation_config[dim:])
    else:
        rot_mat = get_rot_mat_2d(orientation_config[2])
    local_coords = np.einsum("ij, kj -> ki", rot_mat.T, (new_global_coords - orientation_config[:dim]))
    loss = np.linalg.norm((original_local_coords - local_coords), axis=-1).mean()
    return loss


def find_local_frame(
        new_point_coordinates: np.ndarray,
        init_point_coordinates: np.ndarray,
        init_config: np.ndarray,
        n_neighbor_pts_for_origin: int,
        idx_of_detected: np.ndarray = None,
        return_matrix: bool = False
) -> np.ndarray:
    """
    Given detected points in the new scene, optimize a local frame so that when these new points are projected into such
    frame, the MSE between the result and the init_point_coordinates is minimal.

    Args:
        new_point_coordinates: (Q, dim)
        init_point_coordinates: (Q, dim)
        init_config: for 3d, its position + quaternion [x, y, z, roll, pitch, yaw], for 2d its [x, y, alpha]
        n_neighbor_pts_for_origin: how many number of neighbor to consider to estimate origin
        idx_of_detected: in case of some point not been detected, you can specify the index of detected ones, this also
            means that the new_point_coordinates only contains coordinates of the detected points.
        return_matrix: whether to return the configuration as a transposed tranformation matrix

    Returns:
        new frame configuration, of same format as init_config

    """
    dim = new_point_coordinates.shape[-1]
    new_point_coordinates = new_point_coordinates - new_point_coordinates[0]  # the first element is the origin itself
    init_point_coordinates = init_point_coordinates - init_point_coordinates[0]  # similarly as above
    if idx_of_detected is not None:
        init_point_coordinates = init_point_coordinates[idx_of_detected]

    # use the first n_origin_neighbor_pts to estimate the origin of the local frame, the optimization only takes
    # care of matching the orientation
    # origin = new_point_coordinates[:n_neighbor_pts_for_origin].mean(axis=0)
    origin = np.median(new_point_coordinates[:n_neighbor_pts_for_origin], axis=0)

    def obj(orientation_config):
        return get_loss(dim, origin, orientation_config, new_point_coordinates, init_point_coordinates)

    # res = minimize(obj, init_config[dim:], method='Nelder-Mead', tol=1e-6)
    # return np.concatenate([origin, res.x])
    initial_config = copy.deepcopy(init_config)
    initial_config[:dim] = copy.deepcopy(origin)
    res = minimize(obj, initial_config, method='SLSQP', tol=1e-6)
    final = res.x
    final[:dim] = origin
    if return_matrix:
        if dim == 2:
            rot_mat_T = get_rot_mat_2d(final[-1]).T
        elif dim == 3:
            rot_mat_T = get_rot_mat_3d(final[dim:]).T
        else:
            raise NotImplementedError
        return np.concatenate((rot_mat_T, rot_mat_T.dot(origin).reshape(-1, 1)), axis=-1)  # 2d: (2, 3), 3d: (3, 4)
    else:
        return final


def find_neighbors(
        can_shape_coordinates: np.ndarray = None,
        num_neighbor_pts: int = None
) -> np.ndarray:
    """
    Get the initial status of the local frame on canonical shape and sample some reference points around its origin,
    which will be used in later optimization step to find this local frame again on actual obj shape in the new scene.

    Args:
        can_shape_coordinates: (P, dim) the coordinates of all the sampled points on the object
        num_neighbor_pts: number of neighboring points to be considered

    Returns:
        indices of the `num_neighbor_pts` neighbors to each point (P, num_neighbor_pts)

    """
    return np.argsort(
        np.linalg.norm(np.expand_dims(can_shape_coordinates, 1) - np.expand_dims(can_shape_coordinates, 0), axis=-1),
        axis=-1
    )[:, :num_neighbor_pts]
