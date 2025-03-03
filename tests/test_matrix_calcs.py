"""
tests for functions that return transform matrices
"""

import numpy as np
import numpy.testing as npt
import pytest

import cmm_error_map.data_cmpts as dc

abs_tol = 1e-11


def make_3_points():
    """
    create a dict of cases to test  matrix_from_3_points
    """
    tests = {}
    # XY plane null case
    points = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    exp_mat = np.identity(4)
    tests["XY 0"] = {"points": points, "expected": exp_mat}
    # shift XY
    tmat = dc.matrix_from_vectors(vloc=[1.0, 2.0, 3.0], vrot=[0.0, 0.0, 0.0])
    points = tmat @ points
    exp_mat = np.identity(4)
    exp_mat[:3, 3] = [-1.0, -2.0, -3.0]
    tests["XY shift"] = {"points": points, "expected": exp_mat}

    # rotate Z, shift XY
    tmat = dc.matrix_from_vectors(vloc=[1.0, 2.0, 3.0], vrot=[0.0, 0.0, 10.0])
    points = tmat @ points
    tests["XY rotZ"] = {"points": points, "expected": None}

    # rotX
    tmat = dc.matrix_from_vectors(vloc=[1.0, 2.0, 3.0], vrot=[89.0, 0.0, 0.0])
    points = tmat @ points
    tests["rotX"] = {"points": points, "expected": None}

    # rotY
    tmat = dc.matrix_from_vectors(vloc=[1.0, 2.0, 3.0], vrot=[0.0, 32.0, 0.0])
    points = tmat @ points
    tests["rotY"] = {"points": points, "expected": None}

    # small random shifts
    rng = np.random.default_rng()
    points[:3, :] = points[:3, :] + rng.normal(0.0, 0.2, points[:3, :].shape)
    tmat = dc.matrix_from_vectors(vloc=[1.2, -2.1, 3.3], vrot=[10.0, 32.0, -5.0])
    points = tmat @ points
    tests["random"] = {"points": points, "expected": None}

    return tests


pts_tests = make_3_points()


@pytest.mark.parametrize("test_pts", pts_tests.values(), ids=pts_tests.keys())
def test_matrix_from_3_points(test_pts):
    """
    assumes test_pts["points"] is shape 3, 4 and the points are in a rough square
    2 3
    0 1
    """
    pts0 = test_pts["points"]
    calc_mat = dc.matrix_from_3_points(pts0, [0, 1, 2])
    # check if expected matrix is given it matches with calc_mat
    if test_pts["expected"] is not None:
        npt.assert_allclose(calc_mat, test_pts["expected"], atol=abs_tol)

    # check rotation matrix part is valid
    rot_mat = calc_mat[:3, :3]
    iden = rot_mat.T @ rot_mat
    npt.assert_allclose(iden, np.identity(3), atol=abs_tol)
    det = np.linalg.det(rot_mat)
    npt.assert_allclose(det, 1.0, atol=abs_tol)

    # calculate points in new csy
    if pts0.shape[0] == 3:
        pts0 = np.vstack((pts0, np.ones((1, pts0.shape[1]))))

    # check corners are zeroed in new csy
    pts1 = calc_mat @ pts0
    npt.assert_allclose(pts1[:3, 0], np.array([0.0, 0.0, 0.0]), atol=abs_tol)
    npt.assert_allclose(pts1[1:3, 1], np.array([0.0, 0.0]), atol=abs_tol)
    npt.assert_allclose(pts1[2, 2], 0.0, atol=abs_tol)

    # check lengths and angles remain constant
    # calculate lengths 0->1, 0->2, 0->3
    for i in range(3):
        len_0 = np.linalg.norm(pts0[:3, i] - pts0[:3, 0])
        len_1 = np.linalg.norm(pts1[:3, i] - pts1[:3, 0])
        npt.assert_allclose(len_0, len_1)
    # compare angles between 1-> 0 -> 3 and 3 -> 0 -> 2
    # because the lengths have been shown to be the same
    # just need to compare dot products
    v01_0 = pts0[:3, 1] - pts0[:3, 0]
    v02_0 = pts0[:3, 2] - pts0[:3, 0]
    v03_0 = pts0[:3, 3] - pts0[:3, 0]

    v01_1 = pts1[:3, 1] - pts1[:3, 0]
    v02_1 = pts1[:3, 2] - pts1[:3, 0]
    v03_1 = pts1[:3, 3] - pts1[:3, 0]

    a0 = np.dot(v01_0, v03_0)
    a1 = np.dot(v01_1, v03_1)
    npt.assert_allclose(a0, a1, atol=abs_tol)

    b0 = np.dot(v02_0, v03_0)
    b1 = np.dot(v02_1, v03_1)
    npt.assert_allclose(b0, b1, atol=abs_tol)


"""
gui_cmpts.matrix_from_vectors and gui_cmpts.matrix_to_vectors
now use functions from scipy.spatial_transform
so need little testing
"""


def test_matrix_to_vectors_vector_eq():
    # reciprocal random test on vector equivalence
    rng = np.random.default_rng()
    vloc = rng.uniform(-600.0, 600.0, (3,))
    # only tetsing narrow range because of angle equivalence problems
    vrot = rng.uniform(-90, 90.0, (3,))
    mat = dc.matrix_from_vectors(vloc, vrot)
    vloc_calc, vrot_calc = dc.matrix_to_vectors(mat)
    npt.assert_allclose(vloc, vloc_calc, atol=1e-12)
    npt.assert_allclose(vrot, vrot_calc, atol=1e-12)


def test_matrix_to_vectors_matrix_eq():
    # reciprocal random test on matrix equivalence
    rng = np.random.default_rng()
    vloc = rng.uniform(-600.0, 600.0, (3,))
    # test full range here
    vrot = rng.uniform(-720, 720.0, (3,))
    mat = dc.matrix_from_vectors(vloc, vrot)
    vloc_calc, vrot_calc = dc.matrix_to_vectors(mat)
    # go back to matrix
    mat_calc = dc.matrix_from_vectors(vloc_calc, vrot_calc)
    npt.assert_allclose(mat, mat_calc, atol=1e-12)
