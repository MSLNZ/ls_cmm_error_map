"""
trying to locate some bugs
"""

from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg

import pyqtgraph.Qt.QtGui as qtg
import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.Qt.QtCore import Qt as qtc


import qdarktheme

import cmm_error_map.gui_cmpts as gc
import cmm_error_map.data_cmpts as dc
import cmm_error_map.design_matrix_linear_fixed as design


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.machine = dc.pmm_866
        p0 = dc.Probe(title="P0", length=qtg.QVector3D(0, 0, 0))
        p1 = dc.Probe(title="P1", length=qtg.QVector3D(100, 100, -200))
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
        # self.machine.model_params["Rxz"] = 3e-8
        m1.recalculate(self.machine.model_params)
        m2.recalculate(self.machine.model_params)
        self.machine.measurements = {"m1": m1, "m2": m2}

        self.plot2d_docks = []

        self.setup_gui()
        self.make_dock()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.slider_group = self.make_model_sliders()

        self.control_group = Parameter.create(type="group", name="main_controls")
        self.control_group.addChild(self.slider_group)

        self.control_tree = ParameterTree(showHeader=False)
        self.control_tree.setContentsMargins(0, 0, 0, 0)
        self.control_tree.header().setStretchLastSection(True)
        self.control_tree.header().setMinimumSectionSize(100)

        self.control_tree.setParameters(self.control_group, showTop=False)

        v_split = qtw.QSplitter(qtc.Vertical)
        v_split.addWidget(self.control_tree)
        v_split.addWidget(self.summary)
        v_split.setSizes([200, 100])

        h_split = qtw.QSplitter(qtc.Horizontal)
        h_split.addWidget(v_split)
        h_split.addWidget(self.dock_area)
        h_split.setSizes([150, 600])

        layout1 = qtw.QHBoxLayout()
        layout1.addWidget(h_split)
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.setSpacing(20)

        widget = qtw.QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)

    def make_dock(self):
        """
        add a Plot2dDock
        """

        plot_dock = gc.Plot2dDock("Plot", self.machine)
        plot_dock.mmts_to_plot.setValue(["m1"])
        plot_dock.replot()

        self.dock_area.addDock(plot_dock)
        self.plot2d_docks.append(plot_dock)

    def make_model_sliders(self) -> Parameter:
        # create sliders
        slider_group = Parameter.create(
            type="group", title="Linear Model", name="linear_model"
        )
        x_axis = slider_group.addChild(
            dict(type="group", name="X axis", expanded=False)
        )
        y_axis = slider_group.addChild(
            dict(type="group", name="Y axis", expanded=False)
        )
        z_axis = slider_group.addChild(
            dict(type="group", name="Z axis", expanded=False)
        )
        squareness = slider_group.addChild(
            dict(type="group", name="Squareness", expanded=False)
        )
        axes = [x_axis, y_axis, z_axis, squareness]

        for p in design.modelparameters:
            # the 2nd entry in the modelparmeters is the dependent axis
            axes[p[1]].addChild(
                dict(
                    type="slider",
                    name=p[0],
                    title=p[0],
                    limits=[-5.0, 5.0],
                    step=0.1,
                    value=0,
                )
            )

        slider_group.addChild(
            dict(type="action", name="btn_reset_all", title="Reset Model")
        )

        slider_group.sigTreeStateChanged.connect(self.update_model)
        return slider_group

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        control_name = changes[0][0].name()
        slider_value = changes[0][2]
        if control_name in design.model_parameters_dict.keys():
            slider_factor = gc.slider_factors[control_name]
            self.machine.model_params[control_name] = slider_value * slider_factor
        elif control_name == "btn_reset_all":
            self.machine.model_params = design.model_parameters_dict.copy()
            # update sliders
            with self.slider_group.treeChangeBlocker():
                for axis_group in self.slider_group.child("linear_model").children():
                    for child in axis_group.children():
                        child.setValue(0.0)
        self.replot()

    def recalculate(self):
        self.machine.recalculate()

    def replot(self):
        self.recalculate()
        for dock in self.plot2d_docks:
            dock.replot()


def main():
    _app = pg.mkQApp("Bug Hunting in Plot2dDock")
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
