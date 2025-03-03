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

import csv
import datetime as dt
from dataclasses import dataclass, astuple, fields
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

import cmm_error_map.design_poly as design

model_parameters_zero = {
    "Txx": [0.0, 0.0],
    "Txy": [0.0, 0.0],
    "Txz": [0.0, 0.0],
    "Tyx": [0.0, 0.0],
    "Tyy": [0.0, 0.0],
    "Tyz": [0.0, 0.0],
    "Tzx": [0.0, 0.0],
    "Tzy": [0.0, 0.0],
    "Tzz": [0.0, 0.0],
    "Rxx": [0.0, 0.0],
    "Rxy": [0.0, 0.0],
    "Rxz": [0.0, 0.0],
    "Ryx": [0.0, 0.0],
    "Ryy": [0.0, 0.0],
    "Ryz": [0.0, 0.0],
    "Rzx": [0.0, 0.0],
    "Rzy": [0.0, 0.0],
    "Rzz": [0.0, 0.0],
    "Wxy": [0.0, 0.0],
    "Wxz": [0.0, 0.0],
    "Wyz": [0.0, 0.0],
}

# some non-zero parmeters for quick tests
model_parameters_test = {
    "Txx": [0.0, 1.33e-05],
    "Txy": [0.0, 0.0],
    "Txz": [0.0, 0.0],
    "Tyx": [0.0, -1.12e-05],
    "Tyy": [0.0, -5.09e-06],
    "Tyz": [0.0, 0.0],
    "Tzx": [0.0, 2.6e-05],
    "Tzy": [0.0, 4.6e-06],
    "Tzz": [0.0, 3.34e-08],
    "Rxx": [0.0, 7.49e-09],
    "Rxy": [0.0, 1.54e-08],
    "Rxz": [0.0, 5e-09],
    "Ryx": [0.0, -4.58e-09],
    "Ryy": [0.0, -1.43e-08],
    "Ryz": [0.0, 2.19e-08],
    "Rzx": [0.0, 2.49e-09],
    "Rzy": [0.0, -7.94e-10],
    "Rzz": [0.0, 4.78e-08],
    "Wxy": [0.0, 0.0],
    "Wxz": [0.0, 0.0],
    "Wyz": [0.0, 0.0],
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

    def __eq__(self, other):
        # required for testing
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.title != other.title:
            return False
        if self.name != other.name:
            return False
        return np.allclose(self.length, other.length, atol=1e-8)


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

    def recalculate(
        self, model_params: dict[str, float | list[float]], cmm_model: MachineType
    ):
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

        dev3d = design.model_matrix(
            self.grid_nominal,
            self.probe.length,
            model_params,
            cmm_model.fixed_table,
            cmm_model.bridge_axis,
        )
        self.grid_dev = dev3d


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
    fixed: bool = False  # if True, data does not change with model parameters

    def recalculate(
        self, model_params: dict[str, float | list[float]], cmm_model: MachineType
    ):
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
        cmm_dev = design.model_matrix(
            self.cmm_nominal,
            self.probe.length,
            model_params,
            cmm_model.fixed_table,
            cmm_model.bridge_axis,
        )
        self.cmm_dev = cmm_dev

        if ny == 1:
            mmt_deform = cmm_to_bar_csy(self)
        else:
            mmt_deform = cmm_to_plate_csy(self)

        self.mmt_dev = mmt_deform - self.mmt_nominal

    def __eq__(self, other):
        # required for testing, allows mmt1 == mmt2 to be evaluated
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return NotImplemented

        equals = []
        # only make a tuple at the Measurement class level
        t1 = tuple(getattr(self, field.name) for field in fields(self))
        t2 = tuple(getattr(other, field.name) for field in fields(other))
        for a1, a2 in zip(t1, t2):
            if type(a1) is not type(a2):
                equals.append(False)
            elif isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
                equals.append(np.allclose(a1, a2, atol=1e-5))
            else:
                # will use Probe.__eq__ etc.
                equals.append(a1 == a2)
        return all(equals)


def matrix_from_vectors(vloc, vrot):
    """
    takes the vectors from the gui elements (rot in degrees) and
    returns a 4x4 transform matrix
    uses scipy.spatial.transform to convert from gui Euler angles
    """
    rot_st = st.Rotation.from_euler("ZYX", np.flip(vrot), degrees=True)
    mat_st = rot_st.as_matrix()
    loc = np.array(vloc).reshape((-1, 1))
    mat = np.hstack((mat_st, loc))
    transform_mat = np.vstack((mat, np.array([[0, 0, 0, 1]])))
    return transform_mat


def matrix_to_vectors(transform_mat):
    """
    takes a 4 x 4 transformation matrix and returns the Euler angles (rot in degrees).
    """
    vloc = transform_mat[:3, 3]
    rot_st = st.Rotation.from_matrix(transform_mat[:3, :3])
    eul_st = rot_st.as_euler("ZYX", degrees=True)
    vrot = np.flip(eul_st)
    return vloc, vrot


def matrix_from_3_points(
    points: np.ndarray,  # (3 or 4, n)
    corner_inds: list[int],
):
    """
    takes an array of points and returns the matrix (4 x 4) that will transform the points
    to a CSY with
    origin at the point corner_inds[0]
    the x-axis through corner_inds[0]] and corner_inds[1]
    and the z-plane  corner_inds[0], corner_inds[1] and corner_inds[2]
    where corner_inds are the indicies within points
    This is a specific case of a point line plane alignment
    """
    xyz0 = points[:3, corner_inds[0]]
    vx = points[:3, corner_inds[1]] - xyz0
    vy = points[:3, corner_inds[2]] - xyz0

    vz = np.cross(vx, vy)
    vy = np.cross(vz, vx)

    vx = vx / np.linalg.norm(vx)
    vy = vy / np.linalg.norm(vy)
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

    return inv_mat


def cmm_to_plate_csy(mmt: Measurement):
    """
    transform cmm_deform to plate CSY

    origin point ball 0
    z-plane through balls 0, 4, 20
    x-axis through balls 0, 4

    indexes based on mmt.artefact.nballs
    """
    cmm_deform = mmt.cmm_nominal + mmt.cmm_dev
    nx, ny = mmt.artefact.nballs
    xindex = nx - 1
    yindex = nx * (ny - 1)
    corner_inds = [0, xindex, yindex]
    mat = matrix_from_3_points(cmm_deform, corner_inds)
    cmm_deform1 = np.vstack((cmm_deform, np.ones((1, nx * ny))))
    mmt_deform = mat @ cmm_deform1

    return mmt_deform[:3, :]


def cmm_to_bar_csy(mmt: Measurement):
    """
    transform cmm_deform to bar CSY

    origin ball 0
    x-axis through balls 0, nx-1
    z-plane is through balls 0, nx-1, and ?
    """
    cmm_deform = mmt.cmm_nominal + mmt.cmm_dev
    xyz0 = cmm_deform[:, 0]
    xyz1 = cmm_deform[:, -1]

    # for third point
    # take  (0, 100, 0) in bar csy and
    # transform it into cmm csy using mmt.transform_mat
    bar_c2 = np.array([0, 100, 0, 1])
    xyz2 = mmt.transform_mat @ bar_c2
    xyz2 = xyz2[:3]
    corners = np.vstack((xyz0, xyz1, xyz2)).T
    corner_inds = [0, 1, 2]
    mat = matrix_from_3_points(corners, corner_inds)
    cmm_deform1 = np.vstack((cmm_deform, np.ones((1, cmm_deform.shape[1]))))
    mmt_deform = mat @ cmm_deform1

    return mmt_deform[:3, :]


@dataclass
class Machine:
    cmm_model: MachineType
    boxes: dict[str, BoxGrid]
    measurements: dict[str, Measurement]
    probes: dict[str, Probe]
    model_params: dict[str, list[float]]

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
    model_params=model_parameters_zero.copy(),
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
    model_params=model_parameters_zero.copy(),
)


# these could be methods of Measurement class


def short_header(mmt: Measurement, now: str):
    header = ""
    header += f"save time,{now}\n"
    header += f"title,{mmt.title}\n"
    header += f"name,{mmt.name}\n"
    header += f"artefact.title,{mmt.artefact.title}\n"
    header += f"artefact.nballs,{mmt.artefact.nballs[0]},{mmt.artefact.nballs[1]} \n"
    header += f"artefact.ball_spacing,{mmt.artefact.ball_spacing}\n"
    vloc, vrot = matrix_to_vectors(mmt.transform_mat)
    header += f"location,{vloc[0]}, {vloc[1]}, {vloc[2]}\n"
    header += f"rotation/deg,{vrot[0]}, {vrot[1]}, {vrot[2]}\n"
    return header


def mmt_snapshot_to_csv(fp: Path, mmt: Measurement, now: str):
    """
    the minimum data to save for reimporting
    """
    header = short_header(mmt, now)
    header += "id,mmt_x,mmt_y,mmt_z\n"
    np_out = mmt.mmt_nominal + mmt.mmt_dev
    np_out = np.vstack((np.arange(np_out.shape[1]), np_out))
    with open(fp, "w") as fp:
        fp.write(header)
        np.savetxt(fp, np_out.T, delimiter=",", fmt=["%d"] + ["%.5f"] * 3)


def mmt_full_data_to_csv(fp: Path, mmt: Measurement, now: str):
    """
    simulation in cmm and artefact csy
    """
    header = short_header(mmt, now)
    header += "id,"
    header += "cmm_nom_x,cmm_nom_y,cmm_nom_z,"
    header += "cmm_dev_x,cmm_dev_y,cmm_dev_z,"
    header += "mmt_nom_x,mmt_nom_y,mmt_nom_z,"
    header += "mmt_dev_x,mmt_dev_y,mmt_dev_z,"
    header += "\n"

    np_out = np.vstack((mmt.cmm_nominal, mmt.cmm_dev, mmt.mmt_nominal, mmt.mmt_dev))
    np_out = np.vstack((np.arange(np_out.shape[1]), np_out))
    with open(fp, "w") as fp:
        fp.write(header)
        np.savetxt(fp, np_out.T, delimiter=",", fmt=["%d"] + ["%.5f"] * 12)


def mmt_metadata_to_csv(fp: Path, mmt: Measurement, machine: Machine, now: str):
    """
    file with model parameters etc
    """
    header = short_header(mmt, now)
    cmm = machine.cmm_model
    header += f"cmm_model.title,{cmm.title}\n"
    header += f"cmm_model.size,{cmm.size[0]},{cmm.size[1]},{cmm.size[2]}\n"
    header += f"cmm_model.fixed_table,{cmm.fixed_table}\n"
    header += f"cmm_model.bridge_axis,{cmm.bridge_axis}\n"
    prb = mmt.probe
    header += f"probe.title,{prb.title}\n"
    header += f"probe.name,{prb.name}\n"
    header += f"probe.length,{prb.length[0]},{prb.length[1]},{prb.length[2]}\n"
    header += "model parameters\n"
    for key, value in machine.model_params.items():
        header += f"{key},{value}\n"
    with open(fp, "w") as fp:
        fp.write(header)


def mmt_from_snapshot_csv(fn: Path) -> Measurement:
    """
    reads in file either created from `mmt_snapshot_to_csv`
    or created from real measurements
    file should have header and structure as written by `mmt_snapshot_to_csv`
    """
    p0 = Probe(title="P0", name="p0", length=np.array([0, 0, 0]))
    ss_dict = {}
    mmtxyz = []
    with open(fn) as fp:
        snapshot = csv.reader(fp, delimiter=",")
        for row in snapshot:
            if row[0] == "id":
                break
            ss_dict[row[0]] = row[1:]
        for row in snapshot:
            mmtxyz.append([float(row[1]), float(row[2]), float(row[3])])
    mmtxyz = np.array(mmtxyz).T
    artefact = ArtefactType(
        title=ss_dict["artefact.title"][0],
        nballs=[int(ss_dict["artefact.nballs"][0]), int(ss_dict["artefact.nballs"][1])],
        ball_spacing=float(ss_dict["artefact.ball_spacing"][0]),
    )

    vloc = [float(s) for s in ss_dict["location"]]
    vrot = [float(s) for s in ss_dict["rotation/deg"]]

    # nominal position of plate/bar in artefact CSY
    nx, ny = artefact.nballs
    ball_range = np.arange(nx * ny)
    x = (ball_range) % nx * artefact.ball_spacing
    y = (ball_range) // nx * artefact.ball_spacing
    z = (ball_range) * 0.0
    mmt_nominal = np.vstack((x, y, z))
    mmt_nominal1 = np.vstack((x, y, z, np.ones((1, x.shape[0]))))

    mmt_dev = mmtxyz - mmt_nominal
    # nominal position of plate in CMM CSY
    transform_mat = matrix_from_vectors(vloc, vrot)
    cmm_nominal1 = transform_mat @ mmt_nominal1
    cmm_nominal = cmm_nominal1[:3, :]
    cmm_dev = np.zeros_like(cmm_nominal)

    mmt = Measurement(
        title=ss_dict["title"][0],
        name=ss_dict["name"][0],
        artefact=artefact,
        transform_mat=np.identity(4),
        probe=p0,
        cmm_nominal=cmm_nominal,
        cmm_dev=cmm_dev,
        mmt_nominal=mmt_nominal,
        mmt_dev=mmt_dev,
        fixed=True,
    )

    return mmt
