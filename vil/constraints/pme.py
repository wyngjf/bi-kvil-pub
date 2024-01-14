import os
import logging
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numpy.linalg import norm
from scipy import stats as st
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
from sklearn.metrics import pairwise_distances
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from functools import partial

from robot_utils import console
from robot_utils.py.visualize import set_3d_equal_auto

minimize_method = "SLSQP"
# minimize_method = "BFGS"
# minimize_method = "Nelder-Mead"
# kmeans2_method = "random"
kmeans2_method = "points"
# kmeans2_method = "matrix"
pinv_rcond = 1.490116e-08

"""
Subsection 1.2, Kernels for minimization in a semi-normed space of Sobolev type
"""


def eta_kernel(t, lamb):
    """ Reproducing Kernels associated with Sobolev space D^{-2}L^2(R^d) """
    if lamb % 2 == 0:
        if norm(t) == 0:
            y = 0
        else:
            y = (norm(t) ** lamb) * np.log(norm(t))
    else:
        y = norm(t) ** lamb
    return y


def compute_eta(idx_tuple, t_opt, t, lamb):
    return eta_kernel(t[idx_tuple[1]] - t_opt[idx_tuple[0]], lamb)


def compute_eta_matrix(t1, t2, lamb, multiprocessing=True):
    l1, l2 = t1.shape[0], t2.shape[0]
    idx = np.stack(np.meshgrid(np.arange(l1), np.arange(l2)), axis=2).reshape(-1, 2)
    if multiprocessing:
        with Pool(os.cpu_count()) as p:
            return np.array(p.map(partial(compute_eta, t_opt=t1, t=t2, lamb=lamb), idx)).reshape(l2, -1)
    else:
        matrix = np.zeros((l2, l1))
        for i in range(l1):
            for j in range(l2):
                matrix[j, i] = eta_kernel(t2[j] - t1[i], lamb)
        return matrix


def projection(x, f, init_guess):
    """ Projection Index function """

    def dd(t):
        return norm(x - f(t))

    return minimize(dd, init_guess, method=minimize_method, tol=1e-6).x


"""
Section 2, High Dimensional Mixture Density Estimation
Subsection 2.1 
When \mu's and \sigma are given, the following function estimates \hat{\theta}'s.
"""


def weight_seq(x_obs, mu, sigma, epsilon=0.001, max_iter=1000):
    """

    Args:
        x_obs: the data set of interest. (n x D): n D-dimensional data points
        mu:  a vector of the knots in a mixture density estimation. N x D matrix of centroid of N cluster
        sigma: the bandwidth of this density estimation.
        epsilon: a predetermined tolerance of the Euclidean distance between thetas in two consecutive steps.
        max_iter: a predetermined upper bound of the number of steps in this iteration.

    Returns:

    """

    n, D = x_obs.shape
    N = mu.shape[0]
    A = np.zeros(shape=(n, N))
    for j in range(N):
        # Note: prob. of x_obs in the j-th kernel: distribution N(mu[j], sigma ** 2)
        A[:, j] = st.multivariate_normal.pdf(x_obs, mu[j], np.eye(D) * sigma ** 2)

    theta_old = np.repeat(1 / N, N)  # The initial guess for weights of each kernel.
    abs_diff = 10 * epsilon  # Absolute value of the difference between "theta.new" and "theta.old".
    count = 0  # Counting the number of steps in this iteration.
    lambda_hat_old = np.array([n, ] + [-1.0] * D)  # The initial guess of the Lagrangian multipliers

    while abs_diff > epsilon and count <= max_iter:  # The iteration for computing desired theta's
        W = A * theta_old  # \theta_j^{(k)} \times \psi_\sigma(x_i-mu_j)    shape (n x N)
        temp = W.sum(axis=-1, keepdims=True) + 1e-8
        # temp = temp + 1e-5 if temp < 1e-5 else temp
        W = W / temp  # W[i,j] is the posterior probability of Z=j|X=x_i, say w_{i,j}(\theta.old).
        w = W.sum(axis=0)  # w[j] = \sum_{i=1}^n w_{i,j} (N, )

        # each element is the overall contribution of each kernel to the observed datapoint

        def f_lambda(lamb):  # This function is for computing Lagrangian multipliers. lamb is a (D+1)-dim vector
            denom_temp = (np.concatenate((np.ones((N, 1)), mu), axis=-1) * lamb).sum(axis=-1)
            # The denominator sequence:
            # (N, D+1) * (D+1, ) -> (N, D+1).sum(-1) -> (N, )
            # \lambda_1+\lambda_2^T \mu_j, j=1,2,...,N.
            num_temp = mu * w.reshape(-1, 1)  # (N, D) * (N, 1) -> (N, D)

            f1 = (w / denom_temp).sum()  # \sum_{j=1}^N \frac{ w_ij }{ \lambda_1+\lambda_2^T \mu_j }  # (N, )
            f2 = (num_temp / denom_temp.reshape(-1, 1)).sum(axis=0)  # (D, )
            f = norm(f1 - 1.0) + norm(f2 - x_obs.mean(axis=0))
            return f

        lambda_hat = minimize(f_lambda, lambda_hat_old, method=minimize_method, tol=1e-6,
                              options=dict(maxiter=1000)).x  # The lagrangian multipliers.

        # We set the Lagrangian multipliers in the previous step as the initial guess in this step.
        theta_new = w / (np.concatenate((np.ones((N, 1)), mu), axis=-1) * lambda_hat).sum(
            axis=-1)  # The new theta's computed from the old theta's
        abs_diff = norm(theta_new - theta_old)  # The Euclidean distance between the old and new theta vectors.

        if np.isnan(abs_diff):
            abs_diff = 0
            theta_new = theta_old  # It helps us avoid "NA trouble".

        # Set the new theta as the old theta for the next iteration step.
        theta_old = np.clip(theta_new, 0, 1)  # pmax() and pmin() guarantee that theta_j's are in [0,1].
        count += 1
        lambda_hat_old = lambda_hat

    return theta_old  # It returns the estimation of weights \theta_j's.  (N, )


"""
Subsection 2.2
Based on the function weight.seq() in Subsection 2.1, we give the following 
high dimensional mixture density estimation function.
"""


def hdmde(x_obs, n_cluster_min, alpha, max_comp):
    """

    Args:
        x_obs: data set of interest, (n X D) matrix
        n_cluster_min: the lower bound of the number of density components
        alpha: confidence level
        max_comp: upper bound of the number of components in the desired mixture density

    Returns:
        theta_hat: is a vector of weights for knots of this mixture density.
        mu: is a vector of knots of this mixture density.
        sigma: is the variance shared by the components of this mixture density.
        label: the labels indicating which cluster each of the obs point belongs to.

    """

    z_alpha = st.norm.ppf(1 - alpha / 2)
    n, D = x_obs.shape
    n_clusters = n_cluster_min

    def estimate_p(n_clusters):
        # TODO I made the init keypoint selection be able to sample with replacement
        #  /home/gao/opt/anaconda3/envs/dl/lib/python3.8/site-packages/scipy/cluster/vq.py
        #  idx = rng.choice(data.shape[0], size=k, replace=True)
        mu, label = kmeans2(x_obs, n_clusters, minit=kmeans2_method, iter=20)  # mu: N x D matrix of centroids

        # The following block estimates \sigma_N.
        non_empty_cluster_idx = []
        sigma_vec = np.zeros(n_clusters)
        cluster_idx_non_empty = 0
        for j in range(n_clusters):
            index_temp = np.where(label == j)[0]
            if len(index_temp) == 0:
                logging.warning("empty cluster")
                continue
            non_empty_cluster_idx.append(j)
            xi_j = x_obs[index_temp]
            label[index_temp] = cluster_idx_non_empty
            sigma_vec[j] = np.square(norm((xi_j - mu[j]), axis=-1)).mean()
            cluster_idx_non_empty += 1

        mu = mu[non_empty_cluster_idx]
        sigma_vec = sigma_vec[non_empty_cluster_idx]
        n_clusters = mu.shape[0]

        # ic(N)
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1, projection='3d')
        # ax.scatter3D(x_obs[:, 0], x_obs[:, 1], x_obs[:, 2], color='k', s=50)
        # size_vec = np.clip(1500 * sigma_vec, 150, 500)
        # ax.scatter3D(mu[:, 0], mu[:, 1], mu[:, 2], color='lime', s=size_vec, edgecolor='k', alpha=0.5)
        # plt.show()
        # # # exit()

        sig = np.sqrt(sigma_vec.mean() / D)
        theta_hat = weight_seq(x_obs, mu, sig)  # It gives an estimation of theta_j's with weight.seq(). (N, 1)

        # The following block gives an approximation to the underlying density function of interest.
        # This estimation is of the form of weights times scaled kernels.
        A = np.zeros(shape=(n, n_clusters))
        for j in range(n_clusters):
            A[:, j] = st.multivariate_normal.pdf(x_obs, mu[j], sig)
        p = (A * theta_hat.flatten()).sum(axis=-1)  # The second/negative term p_N in Delta.hat. (n, )
        return p, mu, sig, label, theta_hat

    p_old, mu, sig, label, theta_hat = estimate_p(n_clusters)
    n_clusters += 1
    while n_clusters <= min(n, max_comp):
        p_new, mu, sig, label, theta_hat = estimate_p(n_clusters)
        delta_hat = p_new - p_old  # Delta.hat
        delta_hat_std = delta_hat.std()
        delta_hat_std = delta_hat_std + 1e-5 if delta_hat_std < 1e-5 else delta_hat_std
        Z_I_N = np.sqrt(n) * delta_hat.mean() / delta_hat_std  # coefficient of variation
        p_old = p_new
        if z_alpha >= Z_I_N >= -z_alpha and not np.isnan(Z_I_N):
            break
        n_clusters += 1

    return theta_hat, mu, sig, label


"""
Section 3, Principal Manifold Estimation
"""


def pme(x_obs: np.ndarray, d, N0=0, tuning_para_seq=np.exp(np.arange(-5, 6)), alpha=0.05, max_comp=100,
        epsilon=0.05, max_iter=100, print_MSDs=True, viz=False, multiprocessing=True):
    """
    Principal Manifold Estimation
    Remark: The larger N0 is, the less time consuming the function is.

    Args:
        x_obs: (n x D) observation matrix
        d: intrinsic dimension of the underlying manifold
        N0: the number of density components, default value is 20*D
        tuning_para_seq: a vector of tuning parameter candidates, its default value is exp((-15:5)).
            If you would like to fit a manifold for a specific lambda, set tuning.prar.seq=c(lambda)
        alpha: is the pre-determined confidence level, which determines the number of the components in a mixture density.
        max_comp: is the upper bound of the number of components in the desired mixture density.
        epsilon: is the tolerance for distance between manifolds f.old and f.new.
        max_iter: is the upper bound of the number of steps in this iteration.
        print_MSDs:

    Returns: sol_opt_x, sol_opt_t, t_opt, label, embedding, projection
        sol_opt_x: linear mapping for data point
        sol_opt_t: linear mapping for intrinsic parameter
        t_opt: optimal intrinsic parameters
        label: the labels of the original data belonging to which cluster
        embedding: the lower dim embedding of the observation
        projection: the projected observation in observation space

    """
    n, D = x_obs.shape
    lamb = 4 - d

    if N0 == 0:
        N0 = int(min(20 * D, 0.2 * n))

    theta_hat, mu, sigma, label = hdmde(x_obs, N0, alpha, max_comp)
    W = np.diag(theta_hat)
    n_clusters = theta_hat.shape[0]  # the final result of N, the number of components

    if viz:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter3D(x_obs[:, 0], x_obs[:, 1], x_obs[:, 2], color='k', s=50)
        ax.scatter3D(mu[:, 0], mu[:, 1], mu[:, 2], color='lime', s=5000 * theta_hat, edgecolor='k', alpha=0.5)
        plt.show()

    similarities = pairwise_distances(mu)
    iso = MDS(n_components=d, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=-1, normalized_stress=False)
    embedding = iso.fit(similarities).embedding_  # (n_clusters, d)

    mse_list = []
    sol_list = []
    emb_list = []
    emb_obs_list = []
    pro_obs_list = []

    for tuning_idx, w in enumerate(tuning_para_seq):
        console.log(f"[yellow]The tuning parameter is lambda[ {tuning_idx} ] = {w:>2.3f}")

        embedding, sol, ssd_new = compute_sol(embedding, w, W, lamb, n_clusters, d, D, mu)
        count = 1
        ssd_ratio = 10 * epsilon

        while epsilon < ssd_ratio <= 5 and count <= (max_iter - 1):
            ssd_old = ssd_new
            embedding, sol, ssd_new = compute_sol(embedding, w, W, lamb, n_clusters, d, D, mu)
            ssd_ratio = np.abs(ssd_new - ssd_old) / ssd_old
            # console.log(f"-- -- SSD.ratio is {round(ssd_ratio, 4)} and this is the {count}-th step of iteration.")
            count += 1

        sol_opt_x, sol_opt_t = sol[:n_clusters], sol[n_clusters:n_clusters + d + 1]
        emb_obs = projection_index(x_obs, embedding[label], sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t, t_opt=embedding,
                                   intrinsic_dim=d)
        projection = mapping(emb_obs, d, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t, t_opt=embedding)
        ssd_new = np.square(norm(x_obs - projection, axis=-1)).sum()

        mse_list.append(ssd_new)
        console.log(f"-- MSD = {ssd_new:>2.3f}")

        sol_list.append(sol)
        emb_list.append(embedding)
        emb_obs_list.append(emb_obs)
        pro_obs_list.append(projection)

        # if better_at_least_once and tuning_idx >= 4:
        if tuning_idx >= 4:
            if mse_list[tuning_idx] > mse_list[tuning_idx - 1] > mse_list[tuning_idx - 2] > mse_list[tuning_idx - 3]:
                break

    # The following chunk gives the f_\lambda with the optimal \lambda.
    mse_list = np.array(mse_list)
    optimal_idx = np.where(mse_list == mse_list.min())[0].min()
    sol_opt = sol_list[optimal_idx]

    if print_MSDs:
        if viz:
            plt.subplot(1, 1, 1)
            plt.scatter(np.log(tuning_para_seq[:len(mse_list)]), mse_list)
            plt.plot(np.log(tuning_para_seq[:len(mse_list)]), mse_list, c="orange")
            plt.vlines(x=np.log(tuning_para_seq[optimal_idx]), ymin=mse_list.min(), ymax=mse_list.max(),
                       colors="darkgreen", linestyles="--")
            plt.xlabel("Log Lambda")
            plt.ylabel("MSD")
        console.log(f">> The optimal tuning parameter is {tuning_para_seq[optimal_idx]} "
                     f"and the MSD of the optimal fit is {mse_list[optimal_idx]}.")

    return sol_opt[:n_clusters], sol_opt[n_clusters:n_clusters + d + 1], emb_list[optimal_idx], label, \
           emb_obs_list[optimal_idx], pro_obs_list[optimal_idx]


def compute_sol(embedding, w, W, lamb, n_clusters, d, D, mu):
    T = np.concatenate((np.ones((n_clusters, 1)), embedding), axis=-1)  # (I, d+1)
    E = compute_eta_matrix(embedding, embedding, lamb)

    M1 = np.concatenate((
        2 * E.dot(W).dot(E) + 2 * w * E,  # (I, I)
        2 * E.dot(W).dot(T),  # (I, d+1)
        T  # (I, d+1)
    ), axis=-1)  # -> (I, I + 2(d+1))
    M2 = np.concatenate((
        2 * T.T.dot(W).dot(E),  # (d+1, I)
        2 * T.T.dot(W).dot(T),  # (d+1, d+1)
        np.zeros((d + 1, d + 1))  # (d+1, d+1)
    ), axis=-1)  # -> (d+1, I + 2(d+1))

    M3 = np.concatenate((
        T.T,  # (d+1, I)
        np.zeros((d + 1, d + 1)),  # (d+1, d+1)
        np.zeros((d + 1, d + 1))  # (d+1, d+1)
    ), axis=-1)  # -> (d+1, I + 2(d+1))
    # The coefficient matrix of the linear equations
    M = np.concatenate((M1, M2, M3), axis=0)  # (I + 2(d+1), I + 2(d+1))

    # The nonhomogeneous term of the linear equations
    b = np.concatenate((
        2 * E.dot(W).dot(mu),  # (I, D)
        2 * T.T.dot(W).dot(mu),  # (d+1, D)
        np.zeros((d + 1, D)),  # (d+1, D)
    ), axis=0)  # -> (I + 2(d+1), D)
    pinv_M = np.linalg.pinv(M, rcond=pinv_rcond)
    sol = pinv_M.dot(b)  # Solve the linear equations # (I + 2(d+1), D)
    # ic(w, W, embedding, E, T, M1, M2, M3, M, mu, b, pinv_M, sol)

    embedding = projection_index(mu, embedding, sol_opt_x=sol[:n_clusters],
                                 sol_opt_t=sol[n_clusters:n_clusters + d + 1], t_opt=embedding, intrinsic_dim=d)
    projection = mapping(embedding, d, sol_opt_x=sol[:n_clusters], sol_opt_t=sol[n_clusters:n_clusters + d + 1],
                         t_opt=embedding)
    ssd_new = np.square(norm(mu - projection, axis=-1)).sum()
    return embedding, sol, ssd_new


def mapping(t, d, sol_opt_x, sol_opt_t, t_opt):
    """

    Args:
        t: (n, d) array, where n is the number of samples
        d: the intrinsic dimension
        sol_opt_x: the linear mapping matrix in terms of clustering mu (the data point)
        sol_opt_t: the linear mapping matrix in terms of augmented intrinsic parameter t
        t_opt: the optimal intrinsic parameters

    Returns: the mapping of of t in observation space
    """
    if d == 1 and len(t.shape) == 1:
        t = t.reshape(-1, 1)
    elif d == 2 and len(t.shape) == 1:
        t = t.reshape(1, -1)
    t_aug = np.concatenate((np.ones((t.shape[0], 1)), t), axis=-1)
    eta = compute_eta_matrix(t1=t_opt, t2=t, lamb=4 - d, multiprocessing=False)
    return np.einsum("ij,ki->kj", sol_opt_x, eta) + np.einsum("ij,ki->kj", sol_opt_t, t_aug)


def projection_index(data_points, t_init_guess, sol_opt_x, sol_opt_t, t_opt, intrinsic_dim=1):
    """

    Args:
        data_points: the (N, D) array of data points in D-dim observation space
        t_init_guess: initial guess of the t, (N, d) array
        sol_opt_x: projection matrix for the data point
        sol_opt_t: projection matrix for the augmented intrinsic parameter
        t_opt: the optimal intrinsic parameter
        intrinsic_dim: the intrinsic dimension

    Returns:

    """
    if t_init_guess.shape[0] != data_points.shape[0] and t_init_guess.shape[0] == 1:
        t_init_guess = np.tile(t_init_guess, (data_points.shape[0], 1))
    t_data_points = np.zeros((data_points.shape[0], intrinsic_dim))
    for i, (d, t) in enumerate(zip(data_points, t_init_guess)):
        t_data_points[i] = projection(d, partial(mapping, d=intrinsic_dim, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t,
                                                 t_opt=t_opt), t)

    return t_data_points


def pme_func(data_points: np.ndarray, intrinsic_dim: int = 1):
    """
    principal manifold estimation
    Args:
        data_points: (N, dim)
        intrinsic_dim: the dimension of the expected principal manifold

    Returns:

    """
    console.rule("run PME")
    max_value = np.abs(data_points).max() * 0.5
    data_points /= max_value
    sol_opt_x, sol_opt_t, t_opt, label, embedding, projection = pme(
        # data_points, d=intrinsic_dim, tuning_para_seq=np.exp(np.arange(-5, 6)), max_comp=data_points.shape[0]-1
        data_points, d=intrinsic_dim, tuning_para_seq=np.exp(np.arange(-3, 3)), max_comp=data_points.shape[0] // 2
    )
    stress_vectors = data_points - projection
    std_stress = np.linalg.norm(stress_vectors, axis=-1).std()
    std_projection = np.linalg.norm(embedding.var(axis=0))

    # w, mu, sig = vis_pme(data_points, sol_opt_x, sol_opt_t, t_opt, label, embedding, projection, intrinsic_dim, True)
    density_data = embedding.reshape((-1, intrinsic_dim))
    theta_hat, mu, sig, label = hdmde(density_data, n_cluster_min=1, alpha=0.1, max_comp=1)
    return sol_opt_x, sol_opt_t, t_opt, std_stress, std_projection, embedding, max_value, theta_hat, mu, sig, projection, stress_vectors


def vis_pme(data_points, sol_opt_x, sol_opt_t, t_opt, label, embedding, projection, d, plot=False):
    # Note: mapping (intrinsic -> observation space)
    t_test = np.arange(-3, 3, 0.05)
    x_test = mapping(t_test, d=d, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t, t_opt=t_opt)
    # ic(t_opt, label, t_opt.shape, label.shape, data_points.shape)

    # Note: optimization based projection index (observation space -> intrinsic)
    mean_data_point = mapping(embedding.mean(keepdims=True), d=d, sol_opt_x=sol_opt_x,
                              sol_opt_t=sol_opt_t, t_opt=t_opt)
    idx_t_larger_than_mean = np.where(t_test > embedding.mean())[0]
    idx_t_smaller_than_mean = np.where(t_test < embedding.mean())[0]

    # Note: compute density on manifold
    density_data = embedding.reshape((-1, d))
    # ic("hdmde ...")
    theta_hat, mu, sig, label = hdmde(density_data, n_cluster_min=1, alpha=0.1, max_comp=1)
    ic(theta_hat, mu, sig, label)
    if d == 1:
        t_test.reshape(-1, 1)
        p = np.zeros_like(t_test)
        prob_derivative = np.zeros_like(t_test)
        for i in range(mu.shape[0]):
            prob_ = st.multivariate_normal.pdf(t_test.reshape(-1, 1), mu[i], sig) * theta_hat[i]
            p += prob_
            prob_derivative -= prob_ * (t_test - mu[i]) / (sig ** 2)
    elif d == 2:
        lim = 4
        x, y = np.mgrid[-lim:lim:31j, -lim:lim:31j]
        test_data = np.stack((x.ravel(), y.ravel())).T
        p = np.zeros(test_data.shape[0])
        prob_derivative = np.zeros_like(test_data)
        for i in range(mu.shape[0]):
            prob_ = st.multivariate_normal.pdf(test_data, mu[i], sig) * theta_hat[i]
            p += prob_
            prob_derivative -= prob_.reshape(-1, 1) * (test_data - mu[i]) / (sig ** 2)
    else:
        raise ValueError
    entropy = - (p * np.log(p)).sum()
    ic(entropy)

    if plot:
        # Note: density fucntion as polygon
        scaling = 5
        density_func = x_test.copy()
        density_func[:, 2] += p * scaling
        density_polygon_larger_than_mean = np.concatenate(
            (density_func[idx_t_larger_than_mean], x_test[idx_t_larger_than_mean][::-1]),
            axis=0)
        density_polygon_smaller_than_mean = np.concatenate(
            (density_func[idx_t_smaller_than_mean], x_test[idx_t_smaller_than_mean][::-1]),
            axis=0)

        # Note plot
        fig = plt.figure(figsize=(16, 9))
        spec = fig.add_gridspec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0, 0], projection="3d")

        # Note plot stress vector
        stress_vectors = np.concatenate((data_points, projection), axis=-1).reshape(
            (-1, 2, 3))
        lc = Line3DCollection(stress_vectors, colors="k", linewidths=2, zorder=-1)
        ax.add_collection(lc)

        # Note plot density polygon
        poly3d_larger_than_mean = [density_polygon_larger_than_mean.tolist()]
        poly3d_smaller_than_mean = [density_polygon_smaller_than_mean.tolist()]
        ax.add_collection3d(
            Poly3DCollection(poly3d_larger_than_mean, facecolors='paleturquoise', linewidths=2,
                             alpha=0.5, edgecolors="turquoise", zorder=-1))
        ax.add_collection3d(
            Poly3DCollection(poly3d_smaller_than_mean, facecolors='paleturquoise', linewidths=2,
                             alpha=0.5, edgecolors="turquoise", zorder=-1))

        # Note: scatter plots, data points, projection, mean
        ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], color="k", s=30)
        ax.scatter(projection[:, 0], projection[:, 1],
                   projection[:, 2], color="limegreen", alpha=0.8, s=150, zorder=1)
        ax.scatter(mean_data_point[:, 0], mean_data_point[:, 1], mean_data_point[:, 2],
                   color="yellow", edgecolor='k', s=850, zorder=10)

        # Note: plot principal curve
        ax.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], color="darkslategrey", zorder=5)

        # Note: set axis
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        set_3d_equal_auto(ax)
        ax.set_axis_off()
        plt.show()
    return theta_hat, mu, sig