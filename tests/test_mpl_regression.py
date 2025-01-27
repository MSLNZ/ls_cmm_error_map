"""
These are regresion tests of new (2024-25) code
against old (2014) matplotlib code

refactor out setup  code later
"""

import numpy as np
import numpy.testing as npt
import pytest

import cmm_error_map.config.config as cf
import cmm_error_map.data_cmpts as dc
import cmm_error_map.mpl_2014.design_matrix_linear as design_old

# not much attention was paid to the sign of the parameters used in the old code
# as the use of that code was to fit the parameters
# and that would work as long as the code was consistent
# these sign corrections are applied to the parameters in the old code
# so the results match the new code
# the pattern is consistent - Txx, Tyy, Tzz have the same sign all others are reversed

abs_tol = 5e-7  # 0.5 nm

sign_corrections = {
    "Txx": 1,
    "Txy": -1,
    "Txz": -1,
    "Tyx": -1,
    "Tyy": 1,
    "Tyz": -1,
    "Tzx": -1,
    "Tzy": -1,
    "Tzz": 1,
    "Rxx": -1,
    "Rxy": -1,
    "Rxz": -1,
    "Ryx": -1,
    "Ryy": -1,
    "Ryz": -1,
    "Rzx": -1,
    "Rzy": -1,
    "Rzz": -1,
    "Wxy": -1,
    "Wxz": -1,
    "Wyz": -1,
}

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


def make_plate_mmts():
    """
    return a dict of dc.Measurement
    for a plate at each
    CMM plane XY, XZ, YZ
    """
    ballspacing = 133.0
    x0, y0, z0 = 250.0, 50.0, 50.0
    # XY plane
    transform_mat_xy = dc.matrix_from_vectors([x0, y0, z0], [0.0, 0.0, 0.0])
    xt, yt, zt = 0.0, 0.0, -243.4852
    prb_xy = dc.Probe(title="P0", name="p0", length=np.array([xt, yt, zt]))
    mmt_xy = dc.Measurement(
        title="Plate XY",
        name="mmt_00",
        artefact=cf.artefact_models["KOBA 0620"],
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
        artefact=cf.artefact_models["KOBA 0620"],
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
        artefact=cf.artefact_models["KOBA 0620"],
        transform_mat=transform_mat_yz,
        probe=prb_yz,
        cmm_nominal=None,
        cmm_dev=None,
        mmt_nominal=None,
        mmt_dev=None,
    )

    mmts = {"mmt XY": mmt_xy, "mmt XZ": mmt_xz, "mmt YZ": mmt_yz}
    return mmts


plate_mmt_tests = make_plate_mmts()


def make_bar_mmts():
    """
    return a dict of dc.Measurement
    for a bar at each
    CMM plane XY, XZ, YZ
    and  diagonals
    """
    bar = dc.ArtefactType(title="bar 300", nballs=(4, 1), ball_spacing=100.0)
    mmts = make_plate_mmts()
    for mmt in mmts.values():
        mmt.artefact = bar
    # add some diagonals
    return mmts


bar_mmt_tests = make_bar_mmts()


@pytest.mark.parametrize("model_params", model_tests.values(), ids=model_tests.keys())
@pytest.mark.parametrize("mmt", plate_mmt_tests.values(), ids=plate_mmt_tests.keys())
def test_XYZ_regression(model_params, mmt):
    """
    regression test a set of model parameters with the given mmt against the old matplotlib code,
    the old code only calculates for a koba 620 plate on a PMM866.
    """
    # new code
    cmm = dc.pmm_866
    mmt.recalculate(model_params, cmm.cmm_model)

    # matplotlib code
    signed_mp = {
        key: sign_corrections[key] * value for key, value in model_params.items()
    }
    params = list(signed_mp.values())
    xt, yt, zt = mmt.probe.length
    dxy, eXYZ, XYZp, XM, plate_nom = design_old.modelled_mmts_XYZ(
        mmt.transform_mat, xt, yt, zt, params, verbose=True
    )
    # testing
    npt.assert_allclose(XM[:-1, :], mmt.cmm_nominal, atol=abs_tol)
    npt.assert_allclose(eXYZ[:-1, :], mmt.cmm_dev, atol=abs_tol)
    npt.assert_allclose(plate_nom, mmt.mmt_nominal[:-1, :], atol=abs_tol)
    npt.assert_allclose(dxy.T, mmt.mmt_dev[:-1, :], atol=abs_tol)
    return


@pytest.mark.parametrize("model_params", model_tests.values(), ids=model_tests.keys())
@pytest.mark.parametrize("mmt", plate_mmt_tests.values(), ids=plate_mmt_tests.keys())
def test_plate_csy(model_params, mmt):
    """
    test that the plate CSY goes through Balls 1, 5, and 21 correctly
    """
    cmm = dc.pmm_866
    mmt.recalculate(model_params, cmm.cmm_model)
    mmt_deform = mmt.mmt_nominal + mmt.mmt_dev
    # Ball 1 is at (0, 0, 0)
    npt.assert_allclose(mmt_deform[:, 0], np.array([0.0, 0.0, 0.0]), atol=abs_tol)
    # Ball 5 is at  (x, 0, 0)
    npt.assert_allclose(mmt_deform[1:, 4], np.array([0.0, 0.0]), atol=abs_tol)
    # Ball 21 is at  (x, y, 0)
    npt.assert_allclose(mmt_deform[2, 20], 0.0, atol=abs_tol)


@pytest.mark.parametrize("model_params", model_tests.values(), ids=model_tests.keys())
@pytest.mark.parametrize("mmt", bar_mmt_tests.values(), ids=bar_mmt_tests.keys())
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
