import numpy as np
from sklearn.decomposition import PCA


def pca(data_points: np.ndarray):
    """
    Principal Component Analysis
    Args:
        data_points: (N, dim)

    Returns: concatenated array of components (row vectors), mean (mu_x, mu_y, mu_z) and explained variance in (x, y, z)
    """
    pca = PCA().fit(data_points)
    if len(pca.mean_) == 3 and len(pca.explained_variance_) == 2:
        explained_variance = np.concatenate((pca.explained_variance_, np.array([0.0])))
    else:
        explained_variance = pca.explained_variance_
    return np.vstack((pca.components_, pca.mean_, explained_variance))
