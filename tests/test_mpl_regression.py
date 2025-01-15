"""
These are regresion tests of new (2024-25) code
against old (2014) matplotlib code

refactor out setup  code later
"""

import numpy as np
import pytest

import cmm_error_map.config.config as cf
import cmm_error_map.data_cmpts as dc
import cmm_error_map.mpl_2014.design_matrix_linear as design_old

# not much attention was paid to the sign of the parameters used in the old code
# as the use of the code was to fit the parameters
# and that would work as long as the code was consistent
# these sign corrections are applied to the parameters in the old code
# so the results match the new code

sign_corrections = {
    "Txx": 1,
    "Txy": -1,
    "Txz": 1,
    "Tyx": 1,
    "Tyy": 1,
    "Tyz": 1,
    "Tzx": 1,
    "Tzy": 1,
    "Tzz": 1,
    "Rxx": 1,
    "Rxy": 1,
    "Rxz": 1,
    "Ryx": 1,
    "Ryy": 1,
    "Ryz": 1,
    "Rzx": 1,
    "Rzy": 1,
    "Rzz": 1,
    "Wxy": 1,
    "Wxz": 1,
    "Wyz": 1,
}

mp = dc.model_parameters_dict


def make_tests():
    tests = []
    for key in mp:
        test = mp.copy()
        test[key] = 1e-6
        tests.append(test)
    return tests


@pytest.mark.parametrize("model_params", make_tests(), ids=mp.keys())
def test_XY(model_params):
    """
    regression test the  a set of parameter with a Koba plate on the XY plane
    against the old matplotlib code
    """
    # XY plane
    x0, y0, z0 = 250.0, 50.0, 50.0
    x0xy, y0xy, z0xy = x0, y0, z0
    transform_mat_xy = np.array(
        [
            [1.0, 0.0, 0.0, x0xy],
            [0.0, 1.0, 0.0, y0xy],
            [0.0, 0.0, 1.0, z0xy],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    xt, yt, zt = 0.0, 0.0, -243.4852

    # new code
    cmm = dc.pmm_866
    prb_xy = dc.Probe(title="P0", name="p0", length=np.array([xt, yt, zt]))

    mmt = dc.Measurement(
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

    mmt.recalculate(model_params, cmm.cmm_model)

    # matplotlib code
    signed_mp = {
        key: sign_corrections[key] * value for key, value in model_params.items()
    }
    params = list(signed_mp.values())
    dxy, eXYZ, XYZp, XM, plate_nom = design_old.modelled_mmts_XYZ(
        transform_mat_xy, xt, yt, zt, params, verbose=True
    )

    np.testing.assert_allclose(XM[:-1, :], mmt.cmm_nominal, atol=1e-6)
    np.testing.assert_allclose(eXYZ[:-1, :], mmt.cmm_dev, atol=1e-6)
    np.testing.assert_allclose(plate_nom, mmt.mmt_nominal[:-1, :], atol=1e-6)
    np.testing.assert_allclose(dxy.T, mmt.mmt_dev[:-1, :], atol=1e-6)
    return
