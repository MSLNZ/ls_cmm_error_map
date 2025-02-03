"""
design_poly.py
2054-02-03

solve the direct equations in matrix format
only simplification is angle approximation in matrices
allow the input to be a `numpy.Polynomial` object
"""

import numpy as np
from numpy.polynomial import Polynomial


def poly_dependency(
    xyz: np.ndarray,
    model_params: dict[str, float | list[float]],
) -> dict[str, list[float]]:
    """
    if model params[key] is a single value return a linear dependent
    create a polynomial 0.0 * value * xyz[dependent_axis], that is
    parameter * axis position value for the  axis the parameter is dependent on
    if model params[key] is a list of values
    calculate a polynomial form these coeficients and return
    the polnominal evaluated at the axis value for the dependent axis
    """
    x, y, z = xyz
    ld_dict = {
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
    # pl = {}
    # for key, value in model_params.items():
    #     if isinstance(value, list):
    #         poly = Polynomial(value)
    #     else:
    #         poly = Polynomial([0.0, value])
    #     pl[key] = poly(ld_dict[key])
    coeffs = {
        key: value if isinstance(value, list) else [0.0, value]
        for key, value in model_params.items()
    }
    pl = {key: Polynomial(coeffs[key])(ld_dict[key]) for key in model_params}
    return pl


def model_matrix(
    xyz3d: np.ndarray,  # (n, 3)
    xyzt: np.ndarray,  # (3,)
    model_params: dict[str, float],
    fixed_table=False,
    bridge_axis=1,
):
    dev3d = np.apply_along_axis(
        func1d=model_point,
        axis=1,
        arr=xyz3d,
        xyzt=xyzt,
        model_params=model_params,
        fixed_table=fixed_table,
        bridge_axis=bridge_axis,
    )
    return dev3d


def model_point(
    xyz_in: np.ndarray,  # (3,)
    xyzt: np.ndarray,  # (3,)
    model_params: dict[str, list[float]],
    fixed_table=False,
    bridge_axis=1,
):
    """ """
    if not fixed_table:
        # table movement is in opposite direction to probe position for a moving table
        dirn = [1, 1, 1]
        table_axis = int(not bridge_axis)
        dirn[table_axis] = -1
        xyz = xyz_in * dirn
    else:
        xyz = xyz_in

    pl = poly_dependency(xyz, model_params)
    # rotation
    # small angle approximations used
    rxl = np.array(
        [
            [1.0, pl["Rxz"], -pl["Rxy"]],
            [-pl["Rxz"], 1.0, pl["Rxx"]],
            [pl["Rxy"], -pl["Rxx"], 1.0],
        ]
    )
    ryl = np.array(
        [
            [1.0, pl["Ryz"], -pl["Ryy"]],
            [-pl["Ryz"], 1.0, pl["Ryx"]],
            [pl["Ryy"], -pl["Ryx"], 1.0],
        ]
    )
    rzl = np.array(
        [
            [1.0, pl["Rzz"], -pl["Rzy"]],
            [-pl["Rzz"], 1.0, pl["Rzx"]],
            [pl["Rzy"], -pl["Rzx"], 1.0],
        ]
    )
    # inverse rotation
    inv_rxl = np.linalg.inv(rxl)
    inv_ryl = np.linalg.inv(ryl)
    inv_rzl = np.linalg.inv(rzl)
    # translation
    x, y, z = xyz
    xl = np.array([x + pl["Txx"], pl["Txy"], pl["Txz"]])
    yl = np.array([pl["Tyx"], y + pl["Tyy"], pl["Tyz"]])
    zl = np.array([pl["Tzx"], pl["Tzy"], z + pl["Tzz"]])
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

    dev3d = w - xyz_in - xyzt

    return dev3d
