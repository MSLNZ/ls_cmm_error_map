"""
These are regresion tests of new (2024-25) code
against old (2014) matplotlib code

refactor out setup  code later
"""

import numpy as np

import cmm_error_map.config.config as cf
import cmm_error_map.data_cmpts as dc
import cmm_error_map.mpl_2014.design_matrix_linear as design_old


def test_XY_txy():
    """
    regression test the Txy parameter with a Koba plate on the XY plane
    against the old matplotlib code
    TODO parameterise this for the general case
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

    txy = 1e-6

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
    model_params = dc.model_parameters_dict.copy()
    model_params["Txy"] = txy
    mmt.recalculate(model_params, cmm.cmm_model)

    # matplotlib code
    params = np.zeros(21)
    params[1] = txy  # TxyL
    dxy, XYZm, XYZp, XM, plate_nom = design_old.modelled_mmts_XYZ(
        transform_mat_xy, xt, yt, zt, params, verbose=True
    )

    # dxy should be equivalent to mmt.mmt_dev
