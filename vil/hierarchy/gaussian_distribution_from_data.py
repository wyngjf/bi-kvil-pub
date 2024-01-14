import numpy as np
from vil.hierarchy.utils import sphere_logarithmic_map, sphere_exponential_map


def compute_frechet_mean(data, nb_iter_max=50, convergence_threshold=1e-6):
    nb_data = data.shape[0]
    data_len = len(data[0])
    # Initialize to the first data point
    mean = data[0]
    data_on_tangent_space = np.zeros_like(data)
    distance_previous_mean = np.inf
    nb_iter = 0

    while distance_previous_mean > convergence_threshold and nb_iter < nb_iter_max:
        previous_mean = mean
        if data_len == 7:
            for n in range(nb_data):
                data_on_tangent_space[n] = position_orientation_logarithmic_map(data[n],mean)[:, 0]
            mean = position_orientation_exponential_map(mean, np.mean(data_on_tangent_space, axis=0))[:, 0]
            distance_previous_mean = np.linalg.norm(position_orientation_logarithmic_map(mean, previous_mean))
        elif data_len == 4:
            for n in range(nb_data):
                data_on_tangent_space[n] = sphere_logarithmic_map(data[n],mean)[:, 0]
            mean = sphere_exponential_map(mean, np.mean(data_on_tangent_space, axis=0))[:, 0]
            distance_previous_mean = np.linalg.norm(sphere_logarithmic_map(mean, previous_mean))
        nb_iter += 1
    return mean


def compute_manifold_covariance(data: np.ndarray, mean, regularization_factor=1e-8):
    nb_data, data_len = data.shape[:2]
    # Project data on the tangent space
    xts = np.zeros_like(data)
    if data_len == 7:
        for n in range(nb_data):
            xts[n, :] = position_orientation_logarithmic_map(data[n],mean )[:, 0]
    elif data_len == 4:
        for n in range(nb_data):
            xts[n, :] = sphere_logarithmic_map(data[n],mean )[:, 0]

    # Compute covariance
    covariance = np.dot(xts.T, xts)
    # covariance = np.dot(xts.T, xts) + np.eye(nb_dim) * regularization_factor
    # In case a small regularization factor is needed

    return covariance


def position_orientation_logarithmic_map(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
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

    u = np.zeros_like(x)

    u[:3] = x0[:3] - x[:3]
    u[3:] = sphere_logarithmic_map(x[3:], x0[3:])

    return u


def position_orientation_exponential_map(u: np.ndarray, x0: np.ndarray) -> np.ndarray:
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

    x = np.zeros_like(u)

    x[:3] = u[:3] + x0[:3]
    x[3:] = sphere_exponential_map(u[3:], x0[3:])

    return x


nb_data = 100
x = np.random.rand(nb_data, 7)
for n in range(nb_data):
    x[n, 3:] = x[n, 3:] / np.linalg.norm(x[n, 3:])

# Compute mean
mean = compute_frechet_mean(x)
covariance = compute_manifold_covariance(x, mean)
