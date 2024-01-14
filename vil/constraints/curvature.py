import numpy as np


def arc_length(spline):
    # Calculate the differences between adjacent points
    ds = np.diff(spline, axis=0)

    # Calculate the Euclidean distance between adjacent points
    dist = np.linalg.norm(ds, axis=1)

    # Calculate the cumulative sum of the distances
    cum_dist = np.cumsum(dist)

    # Prepend a zero to the cumulative distance array
    cum_dist = np.concatenate(([0], cum_dist))

    return cum_dist


def curvature(spline, s):
    # Calculate the first and second derivatives of the spline
    first_deriv = np.gradient(spline, s, axis=0)
    second_deriv = np.gradient(first_deriv, s, axis=0)

    # Calculate the magnitude of the first derivative
    speed = np.linalg.norm(first_deriv, axis=1)

    # Calculate the curvature
    curvature = np.linalg.norm(np.cross(first_deriv, second_deriv), axis=1) / (speed ** 3)

    return curvature


def get_curvature_of_curve(curve):
    return curvature(curve, arc_length(curve))
