"""
main gui for cmm error map app
0oOilL1I| 0123456789
"""

import pyqtgraph as pg

# import pyqtgraph.opengl as gl
import pyqtgraph.Qt.QtWidgets as qtw
import qdarktheme

# from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt.QtCore import Qt as qtc

import cmm_error_map.design_matrix_linear_fixed as design
import cmm_error_map.gui_cmpts as gc


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.model_params = design.model_parameters_dict.copy()

        self.slider_mag = 5000
        # list of added 2d plots
        self.plot2d_docks = []
        self.plot3d_dock = None
        self.setup_gui()
        self.add_startup_docks()
        self.add_summary()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.slider_group = self.make_model_sliders()
        self.probes_group = self.make_probe_controls()
        self.control_group = Parameter.create(type="group", name="main_controls")
        self.control_group.addChild(self.slider_group)
        self.control_group.addChild(self.control_group)

        # other controls
        btn_plot = self.control_group.addChild(
            dict(type="action", name="btn_plot", title="Add Plot Dock")
        )
        btn_plot.sigActivated.connect(self.add_new_plot2d_dock)

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

    def make_model_sliders(self) -> (Parameter, ParameterTree):
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
            dict(type="action", name="btn_reset_all", title="Reset All")
        )

        slider_group.sigTreeStateChanged.connect(self.update_model)

        return slider_group

    def make_probe_controls(self):
        """
        make control group for probes
        """
        probes_group = Parameter.create(
            type="group", title="Probes", name="probes_group", addText="Add Probe"
        )
        self.add_new_probe_group(probes_group)
        self.add_new_probe_group(probes_group)
        probes_group.sigAddNew.connect(self.add_new_probe_grp)
        probes_group.sigTreeStateChanged.connect(self.update_probes)

        return probes_group

    def add_new_probe_grp(self, parent):
        """
        add the controls for a new probe to the side bar
        """
        new_title = f"Probe {len(parent.childs)}"
        grp_params = gc.probe_control_grp.copy()
        grp_params["title"] = new_title
        grp_params["children"][0]["value"] = new_title
        new_grp = parent.addChild(grp_params, autoIncrementName=True)
        new_grp.child("probe_name").sigValueChanged.connect(self.change_probe_name)

    def change_probe_name(self, param):
        """
        event handler for a change in probe name
        """
        param.parent().setOpts(title=param.value())

    def update_probes(self):
        """
        create self.probes_dict from self.probes_group
        call update_probes method of all docks
        """
        self.probe_data = {}
        for probe_child in self.probes_group.children():
            probe_name = probe_child.name()
            probe_vec = gc.child_to_vector(probe_child.child("grp_probe_lengths"))
            self.probe_data[probe_name] = {
                "probe_title": probe_child.title(),
                "probe_vec": probe_vec,
            }

        self.plot3d_dock.update_probes(self.probe_data)
        for dock in self.plot2d_docks:
            dock.update_probes(self.probe_data)

    def add_startup_docks(self):
        """
        add the 3d plot dock
        """
        # self.plotlines3d, self.plot3d = self.make_plot3d()
        # self.plot3d_params, self.plot3d_tree = self.make_3d_plot_controls()
        # self.make_plot_dock(self.plot3d_tree, self.plot3d, "3d Deformation")
        self.plot3d_dock = gc.Plot3dDock("3D Deformation", self.model_params)
        self.dock_area.addDock(self.plot3d_dock)

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        control_name = changes[0][0].name()
        slider_value = changes[0][2]
        if control_name in design.model_parameters_dict.keys():
            slider_factor = gc.slider_factors[control_name]
            self.model_params[control_name] = slider_value * slider_factor
        elif control_name == "slider_mag":
            self.slider_mag = slider_value
        elif control_name == "btn_reset_all":
            self.model_params = design.model_parameters_dict.copy()
            # update sliders
            with self.slider_group.treeChangeBlocker():
                for axis_group in self.slider_group.child("linear_model").children():
                    for child in axis_group.children():
                        child.setValue(0.0)
        try:
            self.plot3d_dock.update_plot(self.model_params)
        except AttributeError:
            # plot not created yet
            pass
        for dock in self.plot2d_docks:
            dock.update_plot(self.model_params)

    def add_new_plot2d_dock(self):
        """
        create a new plot dock
        type of plot etc is set from side bar on dock
        can have lots of these
        """
        new_plot_dock = gc.Plot2dDock(
            "New Dock",
            self.model_params,
            self.plot3d_dock.plot_data,
        )
        self.dock_area.addDock(new_plot_dock, position="bottom")
        self.plot2d_docks.append(new_plot_dock)

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
