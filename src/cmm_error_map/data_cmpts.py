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

import cmm_error_map.design_linear as design

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
    length: np.ndarray  # (3,)


# for drawing the box deformation
@dataclass
class BoxGrid:
    title: str
    name: str
    npts: (int, int, int)
    spacing: (float, float, float)
    probe: Probe
    grid_nominal: np.array  # (3, n)
    grid_dev: np.array  # (3, n)

    def recalculate(self, model_params: dict[str, float], cmm_model: MachineType):
        """
        update data with new model_parameters
        """
        nx, ny, nz = self.npts
        sx, sy, sz = self.spacing
        pnts_range = np.arange(nx * ny * nz)
        x = (pnts_range % nx) * sx
        y = (pnts_range // nx % ny) * sy
        z = (pnts_range // nx // ny) * sz
        self.grid_nominal = np.stack((x, y, z))

        dev3d = design.linear_model_matrix(
            self.grid_nominal.T,
            self.probe.length,
            model_params,
            cmm_model.fixed_table,
            cmm_model.bridge_axis,
        )
        self.grid_dev = dev3d.T


@dataclass
class Measurement:
    title: str
    name: str
    artefact: ArtefactType
    transform_mat: np.ndarray  # (4, 4)
    probe: Probe

    cmm_nominal: np.ndarray  # (3, n) the nominal position of the balls in CMM CSY
    cmm_dev: np.ndarray  # (3, n) the deformed - nominal position in CMM CSY
    mmt_nominal: np.ndarray  # (3, n) the nominal position of the balls in artefact CSY
    mmt_dev: np.ndarray  # (3, n) the deformed - nominal position in artefact CSY
    fixed: bool = False  # True, data does not change with model parameters

    def recalculate(self, model_params: dict[str, float], cmm_model: MachineType):
        """
        update data with new model_parameters
        """
        if self.fixed:
            return
        # nominal position of plate/bar in artefact CSY
        nx, ny = self.artefact.nballs
        ball_range = np.arange(nx * ny)
        x = (ball_range) % nx * self.artefact.ball_spacing
        y = (ball_range) // nx * self.artefact.ball_spacing
        z = (ball_range) * 0.0
        self.mmt_nominal = np.vstack((x, y, z))
        mmt_nominal1 = np.vstack((x, y, z, np.ones((1, x.shape[0]))))

        # nominal position of plate in CMM CSY
        cmm_nominal1 = self.transform_mat @ mmt_nominal1
        self.cmm_nominal = cmm_nominal1[:3, :]

        # deformed position of plate relative to nominal - in CMM CSY
        cmm_dev = design.linear_model_matrix(
            self.cmm_nominal.T,
            self.probe.length,
            model_params,
            cmm_model.fixed_table,
            cmm_model.bridge_axis,
        )
        self.cmm_dev = cmm_dev.T

        cmm_deform = self.cmm_nominal + self.cmm_dev
        if ny == 1:
            mmt_deform = cmm_to_bar_csy(cmm_deform, self.artefact)
        else:
            mmt_deform = cmm_to_plate_csy(cmm_deform, self.artefact)

        self.mmt_dev = mmt_deform - self.mmt_nominal


def cmm_to_plate_csy(cmm_deform: np.ndarray, artefact: ArtefactType):
    """
    transform cmm_deform to plate CSY

    origin point ball 0
    z-plane through balls 0, 4, 20
    x-axis through balls 0, 4
    """
    nx, ny = artefact.nballs
    xindex = nx - 1
    yindex = nx * (ny - 1)
    xyz0 = cmm_deform[:, 0]
    vx = cmm_deform[:, xindex] - xyz0
    vy = cmm_deform[:, yindex] - xyz0

    vx = vx / np.linalg.norm(vx)
    vy = vy / np.linalg.norm(vy)
    vz = np.cross(vx, vy)
    vz = vz / np.linalg.norm(vz)

    mat = np.array(
        [
            [vx[0], vy[0], vz[0], xyz0[0]],
            [vx[1], vy[1], vz[1], xyz0[1]],
            [vx[2], vy[2], vz[2], xyz0[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    inv_mat = np.linalg.inv(mat)
    cmm_deform1 = np.vstack((cmm_deform, np.ones((1, nx * ny))))
    mmt_deform = inv_mat @ cmm_deform1

    return mmt_deform[:3, :]


def cmm_to_bar_csy(cmm_deform: np.ndarray, artefact: ArtefactType):
    """
    transform cmm_deform to bar CSY

    origin ball 0
    x-axis through balls 0, nx-1
    z-plane is through balls 0, nx-1, and ?
    """
    nx, ny = artefact.nballs
    xindex = nx - 1
    xyz0 = cmm_deform[:, 0]
    vx = cmm_deform[:, xindex] - xyz0
    vx = vx / np.linalg.norm(vx)

    # need a 3rd point/dirn not on bar line
    vy = np.array([0, 1, 0])
    # check bar not along vy
    if 1 - np.dot(vx, vy) < 0.001:
        vy = np.array([1, 0, 0])
    vz = np.cross(vx, vy)
    vz = vz / np.linalg.norm(vz)

    mat = np.array(
        [
            [vx[0], vy[0], vz[0], xyz0[0]],
            [vx[1], vy[1], vz[1], xyz0[1]],
            [vx[2], vy[2], vz[2], xyz0[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    inv_mat = np.linalg.inv(mat)

    cmm_deform1 = np.vstack((cmm_deform, np.ones((1, nx * ny))))
    mmt_deform = inv_mat @ cmm_deform1

    return mmt_deform[:3, :]


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
            mmt.recalculate(self.model_params, self.cmm_model)
        for box in self.boxes.values():
            box.recalculate(self.model_params, self.cmm_model)


# defaults


legex574 = Machine(
    cmm_model=MachineType(
        title="Legex574",
        size=(500, 700, 400),
        fixed_table=False,
        bridge_axis=0,
        box_spacing=(100.0, 100.0, 100.0),
    ),
    boxes={},
    measurements={},
    probes={},
    model_params=model_parameters_dict.copy(),
)

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
