"""
geometry functions that will need testing
uses the mathutils library derived from Blender
"""

import mathutils as mu


def matrix_xz_vector(
    origin: mu.Vector, xaxis: mu.Vector, zaxis: mu.Vector
) -> mu.Matrix:
    """
    create a 4x4 transformation matrix to transform an object to origin aligned to xaxis, zaxis
    should do some check about validity of input and output

    """
    # normalise both axis vectors
    xaxis = xaxis.normalized()
    zaxis = zaxis.normalized()
    # calculate the yaxis perpendicular to x and y
    yaxis = xaxis.cross(zaxis)
    rot_mat = mu.Matrix([xaxis, yaxis, zaxis]).transposed()
    mat4 = mu.Matrix.LocRotScale(origin, rot_mat, None)
    return mat4
