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


@dataclass
class MachineType:
    title: str
    size: (float, float, float)
    fixed_table: bool
    bridge_axis: int


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


@dataclass
class Measurement:
    title: str
    name: str
    artefact: ArtefactType
    transform3d: pg.Transform3D
    probe: Probe
    data2d: np.ndarray
    xyz3d: np.ndarray
    dev3d: np.ndarray

    def recalculate(self, model_params):
        """
        update data with new model_parameters
        """
        pars = list(model_params.values())

        RP = self.transform3d.matrix()
        xt, yt, zt = self.probe.length.x(), self.probe.length.y(), self.probe.length.z()

        self.data2d = design.modelled_mmts_XYZ(
            RP,
            xt,
            yt,
            zt,
            pars,
            ballspacing=self.artefact.ball_spacing,
            nballs=self.artefact.nballs,
        )
        self.xyz3d, self.dev3d = design.data_plot_plate_3d(
            self.artefact,
            self.probe.length,
            model_params,
            self.transform3d,
        )


@dataclass
class Machine:
    cmm_model: MachineType
    measurements: dict[str, Measurement]
    probes: dict[str, Probe]
    model_params: dict[str, float]

    def recalculate(self):
        """
        update all measurement data
        """
        for mmt in self.measurements.values():
            mmt.recalculate(self.model_params)


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
    ),
    measurements={},
    probes={},
    model_params=design.model_parameters_dict.copy(),
)
