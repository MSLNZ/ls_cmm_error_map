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
    nballs: (float, float)
    ball_spacing: float


@dataclass
class Probe:
    title: str
    length: qtg.QVector3D


@dataclass
class Measurement:
    artefact: ArtefactType
    transform3d: pg.Transform3D
    probe: Probe
    data: np.ndarray


@dataclass
class Machine:
    cmm_model: MachineType
    measurements: dict[str, Measurement]
    probes: dict[str, Probe]
    model_params: dict[str, float]


# defaults

# pmm_866_type = MachineType(
#     title="PMM866",
#     size=(800, 600, 600),
#     fixed_table=False,
#     bridge_axis=1,
# )

koba_620_type = ArtefactType(
    title="KOBA 0620",
    nballs=(5, 5),
    ball_spacing=133.0,
)

pmm_866 = Machine(
    cmm_model=MachineType(
        title="PMM866",
        size=(800, 600, 600),
        fixed_table=False,
        bridge_axis=1,
    ),
    measurements=[],
    probes=[],
    model_params=design.model_parameters_dict.copy(),
)
