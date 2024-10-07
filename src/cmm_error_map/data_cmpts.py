"""
A **Machine** has
-  machine type which defines
        - size,
        - bridge/table configuration
- list of **Artefacts**
- list of **Probes**
- current deformation parameters (model)

 An **Artefact** has
 - artefact type which defines
         - size
         - ball spacing
         - nballs
 - location and rotation
 - **Probe**
 - position data calculated from model and pararmeters

A **Probe** has
- x,y,z vector
"""

from dataclasses import dataclass

import numpy as np

import pyqtgraph as pg
import pyqtgraph.Qt.QtGui as qtg

import cmm_error_map.design_matrix_linear_fixed as design

model_parameters_dict = {
    "Txx": 0.0,
    "Txy": 0.0,
    "Txz": 0.0,
    "Tyx": 0.0,
    "Tyy": 0.0,
    "Tyz": 0.0,
    "Tzx": 0.0,
    "Tzy": 0.0,
    "Tzz": 0.0,
    "Rxx": 0.0,
    "Rxy": 0.0,
    "Rxz": 0.0,
    "Ryx": 0.0,
    "Ryy": 0.0,
    "Ryz": 0.0,
    "Rzx": 0.0,
    "Rzy": 0.0,
    "Rzz": 0.0,
    "Wxy": 0.0,
    "Wxz": 0.0,
    "Wyz": 0.0,
}


# some non-zero parmeters for quick tests
model_parameters_test = {
    "Txx": 1.33e-05,
    "Txy": 0.0,
    "Txz": 0.0,
    "Tyx": -1.12e-05,
    "Tyy": -5.09e-06,
    "Tyz": 0.0,
    "Tzx": 2.6e-05,
    "Tzy": 4.6e-06,
    "Tzz": 3.34e-08,
    "Rxx": 7.49e-09,
    "Rxy": 1.54e-08,
    "Rxz": 5e-09,
    "Ryx": -4.58e-09,
    "Ryy": -1.43e-08,
    "Ryz": 2.19e-08,
    "Rzx": 2.49e-09,
    "Rzy": -7.94e-10,
    "Rzz": 4.78e-08,
    "Wxy": 0.0,
    "Wxz": 0.0,
    "Wyz": 0.0,
}


@dataclass
class MachineType:
    title: str
    size: (float, float, float)
    fixed_table: bool
    bridge_axis: int
    box_spacing: (float, float, float)


@dataclass
class ArtefactType:
    title: str
    nballs: (int, int)
    ball_spacing: float


@dataclass
class Probe:
    title: str
    name: str
    length: qtg.QVector3D


# for drawing the box deformation
@dataclass
class BoxGrid:
    title: str
    name: str
    npts: (int, int, int)
    spacing: (float, float, float)
    probe: Probe
    xyz3d: np.array  # (n, 3)
    dev3d: np.array  # (n, 3)

    def recalculate(self, model_params: dict[str, float]):
        """
        update data with new model_parameters
        """
        self.xyz3d, self.dev3d = data_plot3d_box(
            self.probe.length,
            model_params,
            self.npts,
            self.spacing,
        )


@dataclass
class Measurement:
    title: str
    name: str
    artefact: ArtefactType
    transform3d: pg.Transform3D
    probe: Probe
    xy2d: np.ndarray  # (n, 2)
    dev2d: np.ndarray  # (n, 2)
    xyz3d: np.ndarray  # (n, 3)
    dev3d: np.ndarray  # (n, 3)

    def recalculate(self, model_params: dict[str, float]):
        """
        update data with new model_parameters
        """
        pars = list(model_params.values())

        RP = self.transform3d.matrix()
        xt, yt, zt = self.probe.length.x(), self.probe.length.y(), self.probe.length.z()

        self.xy2d, self.dev2d = design.modelled_mmts_XYZ(
            RP,
            xt,
            yt,
            zt,
            pars,
            ballspacing=self.artefact.ball_spacing,
            nballs=self.artefact.nballs,
        )
        self.xyz3d, self.dev3d = data_plot3d_plate(
            self.artefact,
            self.probe.length,
            model_params,
            self.transform3d,
        )


@dataclass
class Machine:
    cmm_model: MachineType
    boxes: dict[str, BoxGrid]
    measurements: dict[str, Measurement]
    probes: dict[str, Probe]
    model_params: dict[str, float]

    def recalculate(self):
        """
        update all measurement data
        """
        for mmt in self.measurements.values():
            mmt.recalculate(self.model_params)
        for box in self.boxes.values():
            box.recalculate(self.model_params)


# defaults


default_artefacts = {
    "KOBA 0620": ArtefactType(
        title="KOBA 0620",
        nballs=(5, 5),
        ball_spacing=133.0,
    ),
    "KOBA 0420": ArtefactType(
        title="KOBA 0420",
        nballs=(5, 5),
        ball_spacing=83.0,
    ),
    "KOBA 0320": ArtefactType(
        title="KOBA 0320",
        nballs=(5, 5),
        ball_spacing=60.0,
    ),
}


pmm_866 = Machine(
    cmm_model=MachineType(
        title="PMM866",
        size=(800, 600, 600),
        fixed_table=False,
        bridge_axis=1,
        box_spacing=(200.0, 200.0, 200.0),
    ),
    boxes={},
    measurements={},
    probes={},
    model_params=model_parameters_dict.copy(),
)


def data_plot3d_plate(
    artefact: ArtefactType,
    prb_length: qtg.QVector3D,
    model_params: dict[str, float],
    transform3d: pg.Transform3D,
) -> (np.ndarray, np.ndarray):
    """
    calulates the nominal position xyz of the
    plate given by artefact,
    at the position defined by transform3d,
    using probe
    and the deviation from nominal xyz_dev
    for the plate deformed by model_params
    returns 2 (n,3) np.ndarray
    """
    ball_range = np.arange(artefact.nballs[0] * artefact.nballs[1])
    x = (ball_range) % artefact.nballs[0] * artefact.ball_spacing
    y = (ball_range) // artefact.nballs[0] * artefact.ball_spacing
    z = (ball_range) * 0.0
    xyz = np.stack((x, y, z))
    xyz = transform3d.map(xyz)
    xt, yt, zt = prb_length.x(), prb_length.y(), prb_length.z()

    xE, yE, zE = design.model_linear(
        xyz[0, :],
        xyz[1, :],
        xyz[2, :],
        list(model_params.values()),
        xt,
        yt,
        zt,
    )
    xyz_dev = np.stack((xE, yE, zE))
    return xyz.T, xyz_dev.T


def data_plot3d_box(
    prb_length: qtg.QVector3D,
    model_params: dict[str, float],
    npts=(5, 4, 4),
    spacing=(200.0, 200.0, 200.0),
) -> (np.ndarray, np.ndarray):
    """
    calulates the nominal position xyz of a box
    with npts on each axis seperated by spacing,
    using probe
    and the deviation from nominal xyz_dev
    for the plate deformed by model_params
    returns 2 (n,3) np.ndarray
    """
    pnts_range = np.arange(npts[0] * npts[1] * npts[2])
    x = (pnts_range % npts[0]) * spacing[0]
    y = (pnts_range // npts[0] % npts[1]) * spacing[1]
    z = (pnts_range // npts[0] // npts[1]) * spacing[2]
    xyz = np.stack((x, y, z))
    xt, yt, zt = prb_length.x(), prb_length.y(), prb_length.z()
    xE, yE, zE = design.model_linear(
        xyz[0, :],
        xyz[1, :],
        xyz[2, :],
        list(model_params.values()),
        xt,
        yt,
        zt,
    )
    xyz_dev = np.stack((xE, yE, zE))
    return xyz.T, xyz_dev.T
