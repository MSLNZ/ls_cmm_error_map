"""
main gui for cmm error map app
"""

import numpy as np
from pyqtgraph.Qt.QtCore import Qt as qtc

import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import pyqtgraph as pg

import pyqtgraph.opengl as gl

import qdarktheme

import cmm_error_map.design_matrix_linear_fixed as design
import cmm_error_map.gui_cmpts as gc


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.model_params = design.model_parameters_dict.copy()
        self.slider_mag = 5000
        self.setup_gui()
        self.make_model_controls()
        self.add_startup_docks()
        self.add_summary()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.slider_params, self.slider_tree = self.make_model_controls()

        v_split = qtw.QSplitter(qtc.Vertical)
        v_split.addWidget(self.slider_tree)
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

    def make_model_controls(self) -> (Parameter, ParameterTree):
        slider_params = Parameter.create(name="params", type="group")
        # create sliders
        slider_group = slider_params.addChild(
            dict(type="group", title="Linear Model", name="linear_model")
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
            dict(type="action", name="btn_reset_all", title="Reset All")
        )

        slider_group.sigTreeStateChanged.connect(self.update_model)
        slider_tree = ParameterTree(showHeader=False)
        slider_tree.setContentsMargins(0, 0, 0, 0)
        slider_tree.header().setStretchLastSection(True)
        slider_tree.header().setMinimumSectionSize(100)
        slider_tree.setParameters(slider_params, showTop=False)
        return slider_params, slider_tree

    def make_3d_plot_controls(self) -> (Parameter, ParameterTree):
        plot3d_params = Parameter.create(name="params", type="group")
        x = np.array([1, 2, 5, 7.5])
        span = np.hstack((x, x * 10, x * 100, x * 1000, [10000]))
        plot_controls = plot3d_params.addChild(
            dict(type="group", name="plot_controls", title="Plot Controls")
        )
        plot_controls.addChild(
            dict(
                type="slider",
                name="slider_mag",
                title="Magnification",
                span=span,
                value=5000,
            )
        )
        plot_controls.sigTreeStateChanged.connect(self.update_model)
        plot3d_tree = ParameterTree(showHeader=False)
        plot3d_tree.setParameters(plot3d_params, showTop=False)
        return plot3d_params, plot3d_tree

    def add_startup_docks(self):
        """
        add the 3d plot dock
        """
        self.plotlines3d, self.plot3d = self.make_plot3d()
        self.plot3d_params, self.plot3d_tree = self.make_3d_plot_controls()
        self.make_plot_dock(self.plot3d_tree, self.plot3d, "3d Deformation")

    def make_plot3d(self) -> (list, gl.GLViewWidget):
        """
        plot the 3d deformation of the CMM volume
        """
        plot3d = gl.GLViewWidget()
        # TODO define CMM size and display spacing (xt, yt, zt)
        xt = 100
        yt = 100
        zt = 100
        # undeformed
        gc.plot_model3d(plot3d, xt, yt, zt, col="green")
        # deformed
        plotlines = gc.plot_model3d(plot3d, xt, yt, zt, col="blue")
        return plotlines, plot3d

    def make_plot_dock(self, tree: ParameterTree, plot: gl.GLViewWidget, title: str):
        """
        add a dock with a parameter tree and a plot
        """
        h_split = qtw.QSplitter(qtc.Horizontal)
        h_split.addWidget(tree)
        h_split.addWidget(plot)
        # h_split.setSizes([150, 600])
        plot_dock = Dock("Plot - side bar")
        plot_dock.addWidget(h_split)
        self.dock_area.addDock(plot_dock)

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        contorl_name = changes[0][0].name()
        slider_value = changes[0][2]
        if contorl_name in design.model_parameters_dict.keys():
            slider_factor = gc.slider_factors[contorl_name]
            self.model_params[contorl_name] = slider_value * slider_factor
        elif contorl_name == "slider_mag":
            self.slider_mag = slider_value
        elif contorl_name == "btn_reset_all":
            self.model_params = design.model_parameters_dict.copy()
            # update sliders
            with self.slider_params.treeChangeBlocker():
                for axis_group in self.slider_params.child("linear_model").children():
                    for child in axis_group.children():
                        child.setValue(0.0)
                        print(f"{child.name()}")

        self.update_plot3d()

    def update_plot3d(self):
        # TODO define CMM size and display spacing (xt, yt, zt)
        xt = 100
        yt = 100
        zt = 100
        gc.update_plot_model3d(
            self.plotlines3d, self.model_params, xt, yt, zt, self.slider_mag
        )

    def add_summary(self):
        text = "Summary\n"
        text += "-------\n\n"
        text += "Use this area for summary or user info"
        self.summary.setPlainText(text)


def main():
    _app = pg.mkQApp("CMM Error Map App")
    qdarktheme.setup_theme(
        "dark",
        custom_colors={"primary": "#f07845", "list.alternateBackground": "#202124"},
    )

    w = MainWindow()
    # app.setStyleSheet("QCheckBox { padding: 5px }")
    w.showMaximized()
    w.show()

    pg.exec()


if __name__ == "__main__":
    main()
