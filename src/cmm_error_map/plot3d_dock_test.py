"""
trying to locate some bugs
"""

import numpy as np
from pyqtgraph.Qt.QtCore import Qt as qtc

import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Qt.QtGui as qtg

import qdarktheme


import cmm_error_map.design_matrix_linear_fixed as design
import cmm_error_map.gui_cmpts as gc
import cmm_error_map.data_cmpts as dc


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_gui()
        self.make_dock()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

    def make_dock(self):
        """
        add a Plot3dDock
        """
        self.machine = dc.pmm_866
        p0 = dc.Probe(title="P0", length=qtg.QVector3D(0, 0, 0))
        p1 = dc.Probe(title="P1", length=qtg.QVector3D(0, 0, -200))
        self.machine.probes = {"p0": p0, "p1": p1}

        m1 = dc.Measurement(
            title="m1",
            artefact=dc.default_artefacts["KOBA 0620"],
            transform3d=pg.Transform3D(),
            probe=p0,
            data=None,
        )
        m2 = dc.Measurement(
            title="m2",
            artefact=dc.default_artefacts["KOBA 0620"],
            transform3d=pg.Transform3D(),
            probe=p1,
            data=None,
        )
        self.machine.model_params["Rxz"] = 3e-8
        m1.recalculate(self.machine.model_params)
        m2.recalculate(self.machine.model_params)
        self.machine.measurements = {"m1": m1, "m2": m2}

        plot_dock = gc.Plot3dDock("Plot", self.machine)

        self.dock_area.addDock(plot_dock)


def main():
    _app = pg.mkQApp("Bug Hunting in Plot3dDock")
    qdarktheme.setup_theme(
        "dark",
        custom_colors={"primary": "#f07845", "list.alternateBackground": "#202124"},
    )

    w = MainWindow()
    w.showMaximized()
    w.show()

    pg.exec()


if __name__ == "__main__":
    main()
