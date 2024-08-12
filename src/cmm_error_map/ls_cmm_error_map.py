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
        self.setup_gui()
        self.add_parameter_tree()
        self.make_docks()
        self.add_summary()

    def setup_gui(self):
        self.dock_area = DockArea()

        self.tree = ParameterTree(showHeader=False)
        self.tree.setContentsMargins(0, 0, 0, 0)
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setMinimumSectionSize(100)

        self.summary = qtw.QTextEdit()

        v_split = qtw.QSplitter(qtc.Vertical)
        v_split.addWidget(self.tree)
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

    def add_parameter_tree(self):
        self.param = Parameter.create(name="params", type="group")
        # create sliders
        slider_group = self.param.addChild(dict(type="group", name="Linear Model"))
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
                    limits=[-5.0, 5.0],
                    step=0.1,
                    value=0,
                )
            )

        slider_group.sigTreeStateChanged.connect(self.update_model)

        self.tree.setParameters(self.param, showTop=False)

    def make_docks(self):
        """
        add a table and a plot in separate docks
        """
        self.table_data, self.table = self.make_table()
        table_dock = Dock("Data Table")
        table_dock.addWidget(self.table)
        self.dock_area.addDock(table_dock)

        self.plotlines3d, self.plot3d = self.make_plot3d()
        plot_dock = Dock("Plot")
        plot_dock.addWidget(self.plot3d)
        self.dock_area.addDock(plot_dock)

    def make_table(self):
        table = pg.TableWidget(editable=False, sortable=True)
        data = np.array(
            [
                (1, 1.6, "x"),
                (3, 5.4, "y"),
                (8, 12.5, "z"),
                (443, 1e-12, "w"),
            ],
            dtype=[("Column 1", int), ("Column 2", float), ("Column 3", object)],
        )
        table.setData(data)
        return data, table

    def make_plot(self):
        plot_data = np.random.normal(size=100)
        plot = pg.plot(plot_data, title="Simplest possible plotting example")
        return plot_data, plot

    def make_plot3d(self):
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

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        slider_name = changes[0][0].name()
        slider_value = changes[0][2]
        slider_factor = gc.slider_factors[slider_name]
        self.model_params[slider_name] = slider_value * slider_factor
        self.update_plot3d()

    def update_plot3d(self):
        # TODO define CMM size and display spacing (xt, yt, zt)
        xt = 100
        yt = 100
        zt = 100
        # TODO add slider for magnification
        mag = 5000
        gc.update_plot_model3d(self.plotlines3d, self.model_params, xt, yt, zt, mag)

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
