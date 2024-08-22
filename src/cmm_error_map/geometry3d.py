"""
geometry functions that will need testing
Coding myself for now, may replace with library
"""

import numpy as np


def matrix_xz_vector(origin: np.ndarray, xaxis: np.ndarray, zaxis: np.ndarray):
    """
    create a 4x4 transformation matrix to transform an object to origin aligned to xaxis, zaxis
    should do some check about validity of input and output

    """
    # normalise both axis vectors
    xaxis = xaxis / np.linalg.norm(xaxis)
    zaxis = xaxis / np.linalg.norm(zaxis)
    # calculate the yaxis perpendicular to x and y
    yaxis = np.linalg.cross(xaxis, zaxis)

    m1 = np.vstack((xaxis, yaxis, zaxis, origin)).T
    m2 = np.vstack((m1, [0, 0, 0, 1]))
    return m2
