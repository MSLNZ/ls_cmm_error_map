"""
simple demo of 2d plot for development
"""

import pyqtgraph as pg
import pyqtgraph.Qt.QtGui as qtg
from cmm_error_map import data_cmpts as dc
from cmm_error_map import gui_cmpts as gc
from cmm_error_map import design_matrix_linear_fixed as design


pg.mkQApp()
w = pg.PlotWidget()
w.setAspectLocked()
p0 = dc.Probe(title="P0", name="p0", length=qtg.QVector3D(0, 0, 0))
mmt = dc.Measurement(
    title="m1",
    name="mmt_00",
    artefact=dc.default_artefacts["KOBA 0620"],
    transform3d=pg.Transform3D(),
    probe=p0,
    xy2d=None,
    dev2d=None,
    xyz3d=None,
    dev3d=None,
)
mmt.artefact.nballs = (5, 4)
model_params = design.model_parameters_dict.copy()
mmt.recalculate(model_params)
balls, lines = gc.plot2d_plate(w, mmt)
w.show()
pg.exec()
