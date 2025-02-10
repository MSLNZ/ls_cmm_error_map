"""
design_poly.py
2054-02-03

solve the direct equations in matrix format
only simplification is angle approximation in matrices
allow the input to be a `numpy.Polynomial` object
"""

import logging

import numpy as np
from numpy.polynomial import Polynomial

logger = logging.getLogger(__name__)


def evaluate_polynomials(
    xyz: np.ndarray,  # (3, n)
    model_params: float | list[float],
):
    """
    create a polynomial for each parameter in model_params
    and evaluate that polynomoal at each point in xyz
    """
    x, y, z = xyz
    dep_dict = {
        "Txx": x,
        "Txy": x,
        "Txz": x,
        "Tyx": y,
        "Tyy": y,
        "Tyz": y,
        "Tzx": z,
        "Tzy": z,
        "Tzz": z,
        "Rxx": x,
        "Rxy": x,
        "Rxz": x,
        "Ryx": y,
        "Ryy": y,
        "Ryz": y,
        "Rzx": z,
        "Rzy": z,
        "Rzz": z,
        "Wxy": x,
        "Wxz": y,
        "Wyz": z,
    }
    # create polys from params
    coeffs = {
        key: value if isinstance(value, list) else [0.0, value]
        for key, value in model_params.items()
    }
    model_polys = {key: Polynomial(coeffs[key]) for key in model_params}
    poly_evals = {}
    for key in dep_dict:
        poly_evals[key] = model_polys[key](dep_dict[key])
    return poly_evals


def model_matrix(
    xyz_in: np.ndarray,  # (3, n)
    xyzt: np.ndarray,  # (3,)
    model_params: dict[str, float | list[float]],
    fixed_table=False,
    bridge_axis=1,
) -> np.ndarray:  # (3, n)
    """
    return the cmm deviation at the nominal points in xyz_in
    for probe lengths xyzt for the model in model_params
    """
    logger.debug("in model_matrix")

    if not fixed_table:
        # table movement is in opposite direction to probe position for a moving table
        dirn = [1, 1, 1]
        table_axis = int(not bridge_axis)
        dirn[table_axis] = -1
        xyz = (xyz_in.T * dirn).T
    else:
        xyz = xyz_in

    # this is slow keep outside loop
    pl = evaluate_polynomials(xyz, model_params)

    dev_out = np.empty_like(xyz_in)
    for i in range(xyz_in.shape[1]):
        # could parameterise loop but its fast enough for now
        rxl = np.array(
            [
                [1.0, pl["Rxz"][i], -pl["Rxy"][i]],
                [-pl["Rxz"][i], 1.0, pl["Rxx"][i]],
                [pl["Rxy"][i], -pl["Rxx"][i], 1.0],
            ]
        )
        ryl = np.array(
            [
                [1.0, pl["Ryz"][i], -pl["Ryy"][i]],
                [-pl["Ryz"][i], 1.0, pl["Ryx"][i]],
                [pl["Ryy"][i], -pl["Ryx"][i], 1.0],
            ]
        )
        rzl = np.array(
            [
                [1.0, pl["Rzz"][i], -pl["Rzy"][i]],
                [-pl["Rzz"][i], 1.0, pl["Rzx"][i]],
                [pl["Rzy"][i], -pl["Rzx"][i], 1.0],
            ]
        )
        # inverse of a rotation matrix is its transpose
        inv_rxl = rxl.T
        inv_ryl = ryl.T
        inv_rzl = rzl.T
        # translation
        x, y, z = xyz[:, i]
        xl = np.array([x + pl["Txx"][i], pl["Txy"][i], pl["Txz"][i]])
        yl = np.array([pl["Tyx"][i], y + pl["Tyy"][i], pl["Tyz"][i]])
        zl = np.array([pl["Tzx"][i], pl["Tzy"][i], z + pl["Tzz"][i]])
        # @ is matrix multiplication, fmt:skip keeps formatting when using ruff
        if fixed_table and bridge_axis == 0:
            # moving bridge, x axis across bridge - eg Hexagon ... (Bruce)
            w = (
                + yl
                + inv_ryl @ xl
                + inv_ryl @ inv_rxl @ zl
                + inv_ryl @ inv_rxl @ inv_rzl @ xyzt
            )  # fmt: skip

        elif fixed_table and bridge_axis == 1:
            # moving bridge y-axis across bridge - eg ? - for completeness
            w = (
                + xl
                + inv_rxl @ yl
                + inv_rxl @ inv_ryl @ zl
                + inv_rxl @ inv_ryl @ inv_rzl @ xyzt
            )  # fmt: skip
        elif not fixed_table and bridge_axis == 0:
            # fixed brige, moving table, x-axis across bridge eg. Mitutoyo  (Shishimi)
            w =  (
                - ryl @ yl 
                + ryl @ xl 
                + ryl @ inv_rxl @ zl 
                + ryl @ inv_rxl @ inv_rzl @ xyzt
            )  # fmt: skip
        elif not fixed_table and bridge_axis == 1:
            # fixed brige, moving table, y-axis across bridge eg. old PMM866
            w = (
                - rxl @ xl 
                + rxl @ yl 
                + rxl @ inv_ryl @ zl 
                + rxl @ inv_ryl @ inv_rzl @ xyzt
            )  # fmt: skip
        else:
            raise ValueError("Unknown cmm type")

        dev3d = w - xyz_in[:, i] - xyzt
        dev_out[:, i] = dev3d

        # using polynomial terms for model parameters can mean there is  non-zero values at the
        # zero position of CMM
        # but the following line gives odd distortion
        # this ony happens in very contrived examples
        # leave commented oiut for now
        # dev_out = dev_out - dev_out[:, 0:1]

    return dev_out
