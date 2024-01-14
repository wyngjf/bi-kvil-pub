import os
import copy
import numpy as np

from typing import Tuple
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.neighbors import LocalOutlierFactor


def get_neighbor_and_inlier_state_matrix(number_of_points: int, inlier_idx: np.ndarray, neighbors: np.ndarray):
    """

    Args:
        number_of_points: the state matrix will have shape ( number_of_points, number_of_points )
        inlier_idx: (n, ) indicating the inlier index of the points, the corresponding column of the
            result matrix will be set to one
        neighbors: ( number_of_points, num_of_neighbor ) indicating the index of neighboring points

    Returns:

    """
    P = number_of_points
    state_matrix = np.full((P, P), False)  # P, P
    # state_matrix[:, j] = True if the j-th point is an inlier
    state_matrix[:, inlier_idx] = True
    mask = np.full((P, P), 1)
    # if a point j is not the neighbor of the point i, mask[i, j] = True,
    mask[np.arange(P)[: ,np.newaxis], neighbors] = 0
    # and if a point i is an inlier, mask[i, :] = True
    mask[inlier_idx] = 1
    # which means, only the state of outlier's neighbor is considered
    masked_state = np.ma.masked_array(state_matrix, mask.tolist(), dtype=bool)
    return masked_state


def get_outlier_idx(data_points: np.ndarray, n_neighbors: int = 40) -> np.ndarray:
    """

    Args:
        data_points: (num_of_point, dimension)
        n_neighbors: how many neighbors are required to determine the outlier state

    Returns: (num_outliers, )

    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    return np.where(clf.fit_predict(data_points) == -1)[0]


def outlier_detection(
        traj: np.ndarray,
        n_neighbors: int = 40,
        exclude_idx: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        traj: (time_steps, num_of_point, dimension) or (num_of_point, dimension)
        n_neighbors: how many neighbors are required to determine the outlier state
        exclude_idx: (num_of_point, ) additional point indices that need to be excluded

    Returns: inlier_idx (n, ) and outlier_idx (n, )

    """

    if np.ndim(traj) == 2:
        P = traj.shape[0]
        outlier_idx_in_all_traj = get_outlier_idx(traj, n_neighbors=n_neighbors)
    else:
        # console.print(f"removing outliers from traj")
        T, P, dim = traj.shape
        with Pool(os.cpu_count()) as p:
            outlier_idx_in_all_traj = np.concatenate(
                p.map(partial(get_outlier_idx, n_neighbors=n_neighbors), traj)
            )

        outlier_idx_in_all_traj = np.unique(outlier_idx_in_all_traj)

    outlier_mask = np.zeros(P, dtype=bool)  # True for outliers
    outlier_mask[outlier_idx_in_all_traj] = True

    if exclude_idx is not None:
        outlier_mask[exclude_idx] = True

    # get inlier index and save to file
    inlier_index = np.where(np.invert(outlier_mask))[0]
    outlier_idx = np.where(outlier_mask)[0]
    return inlier_index, outlier_idx


def fix_depth_for_outlier_in_one_frame(masked_state: np.ma.masked_array, raw_depth: np.ndarray, uv: np.ndarray):
    """

    Args:
        masked_state: (P, P) where P is the total number of points, including inlier and outlier
        raw_depth: (h, w) depth map
        uv: (P, 2)

    Returns: return the fixed depth map (h, w)

    """
    masked_state_at = copy.deepcopy(masked_state)
    fixed_depth_at = copy.deepcopy(raw_depth)

    while np.sum(masked_state_at, axis=-1).data[np.argmax(np.sum(masked_state_at, axis=-1))] > 0:
        # find the outlier point that has the most number of inlier neighbors
        selected_outlier_idx = np.argmax(np.sum(masked_state_at, axis=-1))
        outlier_uv = uv[selected_outlier_idx]

        # the neighbor of the selected outlier that belongs to inlier
        neighbor_idx = np.ma.where(masked_state_at[selected_outlier_idx])[0]
        neighbor_uv = uv[neighbor_idx]

        # use the neighbors' average depth as the depth of the outlier point
        fixed_depth_at[outlier_uv[1], outlier_uv[0]] = np.mean(fixed_depth_at[neighbor_uv[:, 1], neighbor_uv[:, 0]])

        # update the outlier state,
        # and the outlier state of the selected outlier is set to inlier
        masked_state_at[:, selected_outlier_idx] = np.logical_not(masked_state_at[:, selected_outlier_idx])
        # the row of the selected outlier is masked out for next loops
        masked_state_at.mask[selected_outlier_idx] = True

    return fixed_depth_at

