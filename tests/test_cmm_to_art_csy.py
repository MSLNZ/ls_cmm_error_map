"""
test for the functions to transform from deformation
relative to CMM csy to deformation relative to artefact csy
"""

import numpy as np
import numpy.testing as npt
import pytest

import cmm_error_map.data_cmpts as dc

abs_tol = 1e-11

pmax = {
    "Txx": 1e-6,
    "Txy": 1e-6,
    "Txz": 1e-6,
    "Tyx": 1e-6,
    "Tyy": 1e-6,
    "Tyz": 1e-6,
    "Tzx": 1e-6,
    "Tzy": 1e-6,
    "Tzz": 1e-6,
    "Rxx": 1e-8,
    "Rxy": 1e-8,
    "Rxz": 1e-8,
    "Ryx": 1e-8,
    "Ryy": 1e-8,
    "Ryz": 1e-8,
    "Rzx": 1e-8,
    "Rzy": 1e-8,
    "Rzz": 1e-8,
    "Wxy": 1e-8,
    "Wxz": 1e-8,
    "Wyz": 1e-8,
}

mp = dc.model_parameters_dict

keys_to_test = list(mp.keys())[:-3]


def make_models():
    """
    return a dict of copies of dc.model_parameters_dict
    where each copy has one non-zero value
    set to the corresponding value from pmax
    """
    tests = {}
    for key in keys_to_test:
        test = mp.copy()
        test[key] = pmax[key]
        tests[key] = test
    return tests


model_tests = make_models()

# add a model with multiple entries
model_tests["multi"] = dc.model_parameters_test.copy()


def make_bar_mmts():
    """
    return a dict of dc.Measurement
    for a plate at each
    CMM plane XY, XZ, YZ
    """
    bar = dc.ArtefactType(title="bar 300", nballs=(4, 1), ball_spacing=100.0)
    ballspacing = 133.0
    x0, y0, z0 = 250.0, 50.0, 50.0
    # XY plane
    transform_mat_xy = dc.matrix_from_vectors([x0, y0, z0], [0.0, 0.0, 0.0])
    xt, yt, zt = 0.0, 0.0, -243.4852
    prb_xy = dc.Probe(title="P0", name="p0", length=np.array([xt, yt, zt]))
    mmt_xy = dc.Measurement(
        title="Plate XY",
        name="mmt_00",
        artefact=bar,
        transform_mat=transform_mat_xy,
        probe=prb_xy,
        cmm_nominal=None,
        cmm_dev=None,
        mmt_nominal=None,
        mmt_dev=None,
    )
    # XZ plane
    transform_mat_xz = dc.matrix_from_vectors(
        [x0, y0 + 2.0 * ballspacing, z0],
        [90.0, 0.0, 0.0],
    )
    xt_xz, yt_xz, zt_xz = 0.0, 130.0, -243.4852
    prb_xz = dc.Probe(title="P0", name="p0", length=np.array([xt_xz, yt_xz, zt_xz]))
    mmt_xz = dc.Measurement(
        title="Plate XZ",
        name="mmt_01",
        artefact=bar,
        transform_mat=transform_mat_xz,
        probe=prb_xz,
        cmm_nominal=None,
        cmm_dev=None,
        mmt_nominal=None,
        mmt_dev=None,
    )
    # YZ plane
    transform_mat_yz = dc.matrix_from_vectors(
        [x0 + 2.0 * ballspacing, y0, z0],
        [90.0, 0.0, 90.0],
    )
    xt_yz, yt_yz, zt_yz = 130.0, 0.0, -243.4852
    prb_yz = dc.Probe(title="P0", name="p0", length=np.array([xt_yz, yt_yz, zt_yz]))
    mmt_yz = dc.Measurement(
        title="Plate YZ",
        name="mmt_02",
        artefact=bar,
        transform_mat=transform_mat_yz,
        probe=prb_yz,
        cmm_nominal=None,
        cmm_dev=None,
        mmt_nominal=None,
        mmt_dev=None,
    )

    mmts = {"mmt XY": mmt_xy, "mmt XZ": mmt_xz, "mmt YZ": mmt_yz}
    return mmts


mmt_tests = make_bar_mmts()


@pytest.mark.parametrize("model_params", model_tests.values(), ids=model_tests.keys())
@pytest.mark.parametrize("mmt", mmt_tests.values(), ids=mmt_tests.keys())
def test_bar_csy(model_params, mmt):
    """
    test the end points of mmt
    """
    cmm = dc.pmm_866
    mmt.recalculate(model_params, cmm.cmm_model)
    mmt_deform = mmt.mmt_nominal + mmt.mmt_dev
    # Ball 1 is at (0, 0, 0)
    npt.assert_allclose(mmt_deform[:, 0], np.array([0.0, 0.0, 0.0]), atol=abs_tol)
    # end ball is at  (x, 0, 0)
    npt.assert_allclose(mmt_deform[1:, -1], np.array([0.0, 0.0]), atol=abs_tol)
