"""
main gui for cmm error map app
0oOilL1I| 0123456789
"""

import numpy as np
import pyqtgraph as pg

import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.Qt.QtCore import Qt as qtc
# import pyqtgraph.Qt.QtGui as qtg

import qdarktheme

from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree

import cmm_error_map.gui_cmpts as gc
import cmm_error_map.data_cmpts as dc
import cmm_error_map.config.config as cf


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.machine = dc.pmm_866
        self.cmm_models = cf.cmm_models

        # list of added 2d plots
        self.plot2d_docks = []
        self.plot3d_dock = None
        self.setup_gui()
        self.add_startup_docks()
        self.add_summary()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.make_model_sliders()
        self.make_probe_controls()
        self.make_measurement_controls()
        self.make_machine_controls()

        self.control_group = Parameter.create(type="group", name="main_controls")

        self.control_group.addChild(self.machine_group)
        self.control_group.addChild(self.prb_group)
        self.control_group.addChild(self.mmt_group)
        self.control_group.addChild(self.slider_group)

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

    def make_machine_controls(self):
        limits = list(self.cmm_models.keys())
        self.machine_group = Parameter.create(
            type="list",
            title="Machine",
            name="machine",
            limits=limits,
            value=limits[0],
        )
        self.machine_group.sigTreeStateChanged.connect(self.update_machine)

    def update_machine(self):
        self.machine = dc.Machine(
            cmm_model=self.cmm_models[self.machine_group.value()],
            boxes={},
            measurements={},
            probes={},
            model_params={},
        )
        # read model, mesurement, and probes from gui
        axis_children = ["x_axis", "y_axis", "z_axis", "squareness"]
        for control_name, axis_id in gc.axis_group.items():
            slider_factor = gc.slider_factors[control_name]
            axis_group = self.slider_group.child(axis_children[axis_id])
            slider_value = axis_group.child(control_name).value()
            self.machine.model_params[control_name] = slider_value * slider_factor

        # this will call replot twice - optimize if needed
        self.update_probes()
        self.update_measurements()

    def make_model_sliders(self) -> Parameter:
        # create sliders
        self.slider_group = Parameter.create(
            type="group", title="Linear Model", name="linear_model"
        )
        x_axis = self.slider_group.addChild(
            dict(type="group", title="X axis", name="x_axis", expanded=False)
        )
        y_axis = self.slider_group.addChild(
            dict(type="group", title="Y axis", name="y_axis", expanded=False)
        )
        z_axis = self.slider_group.addChild(
            dict(type="group", title="Z axis", name="z_axis", expanded=False)
        )
        squareness = self.slider_group.addChild(
            dict(type="group", title="Squareness", name="squareness", expanded=False)
        )
        axes = [x_axis, y_axis, z_axis, squareness]

        for key, axis in gc.axis_group.items():
            axes[axis].addChild(
                dict(
                    type="slider",
                    name=key,
                    title=key,
                    limits=[-5.0, 5.0],
                    step=0.1,
                    value=0,
                )
            )

        self.slider_group.addChild(
            dict(type="action", name="btn_reset_all", title="Reset Model")
        )

        self.slider_group.sigTreeStateChanged.connect(self.update_model)

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        control_name = changes[0][0].name()
        slider_value = changes[0][2]
        if control_name in dc.model_parameters_dict.keys():
            slider_factor = gc.slider_factors[control_name]
            self.machine.model_params[control_name] = slider_value * slider_factor
        elif control_name == "btn_reset_all":
            self.machine.model_params = dc.model_parameters_dict.copy()
            # update sliders
            with self.slider_group.treeChangeBlocker():
                for axis_group in self.slider_group.children():
                    for child in axis_group.children():
                        child.setValue(0.0)
        self.replot()

    def make_probe_controls(self) -> Parameter:
        """
        make control group for probes
        """
        self.prb_group = Parameter.create(
            type="group",
            title="Probes",
            name="probes_group",
            addText="Add Probe",
        )
        self.add_new_probe_group(self.prb_group)
        self.add_new_probe_group(self.prb_group)
        self.update_probes()
        self.prb_group.sigAddNew.connect(self.add_new_probe_group)
        self.prb_group.sigTreeStateChanged.connect(self.update_probes)

    def add_new_probe_group(self, parent):
        """
        add the controls for a new probe to the side bar
        """
        new_title = f"Probe {len(parent.childs)}"
        grp_params = gc.probe_control_grp.copy()
        new_grp = parent.addChild(grp_params, autoIncrementName=True)
        new_grp.child("prb_title").sigValueChanged.connect(self.change_prb_title)
        new_grp.child("prb_title").setValue(new_title)

    def update_probes(self):
        """
        recreates self.machine.probes from gui controls in self.probes_group
        """
        self.machine.probes = {}
        spacing = self.machine.cmm_model.box_spacing
        size = self.machine.cmm_model.size
        npts = (
            int(size[0] // spacing[0]) + 1,
            int(size[1] // spacing[1]) + 1,
            int(size[2] // spacing[2]) + 1,
        )

        for probe_child in self.prb_group.children():
            probe_name = probe_child.name()
            grp_probe = probe_child.child("grp_probe_lengths")
            vprobe = [grand_kid.value() for grand_kid in grp_probe]
            probe_vec = np.array(vprobe)
            probe = dc.Probe(
                title=probe_child.title(),
                name=probe_child.name(),
                length=probe_vec,
            )
            self.machine.probes[probe_name] = probe
            # also create box deformation object

            box = dc.BoxGrid(
                title=probe_child.title(),
                name=probe_name,
                spacing=spacing,
                npts=npts,
                probe=probe,
                grid_nominal=None,
                grid_dev=None,
            )

            self.machine.boxes[probe_name] = box
        self.replot()

    def make_measurement_controls(self) -> Parameter:
        self.mmt_group = Parameter.create(
            type="group",
            title="Measurements",
            name="mmt_group",
            addText="Add Measurement",
        )
        self.add_new_mmt_group(self.mmt_group)
        self.mmt_group.sigAddNew.connect(self.add_new_mmt_group)
        self.mmt_group.sigTreeStateChanged.connect(self.update_measurements)

    def add_new_mmt_group(self, parent):
        """
        add the controls for a new artefact measurement to the side bar
        """
        new_title = f"Measurement{len(parent.childs)}"
        grp_params = gc.mmt_control_grp.copy()

        with self.mmt_group.treeChangeBlocker():
            new_grp = parent.addChild(grp_params, autoIncrementName=True)
            new_grp.setOpts(title=new_title)
            new_grp.child("mmt_title").setValue(new_title)
            new_grp.child("artefact").setLimits(list(dc.default_artefacts.keys()))
            new_grp.child("artefact").setValue(list(dc.default_artefacts.keys())[0])
            prb_limits = {
                value.title: key for key, value in self.machine.probes.items()
            }
            new_grp.child("probe").setLimits(prb_limits)
            new_grp.child("probe").setValue(list(prb_limits.values())[0])
            new_grp.child("mmt_title").sigValueChanged.connect(self.change_mmt_title)

        self.update_measurements()

    def update_measurements(self):
        """
        recreates self.machine.measurements from gui controls in self.mmt_group
        recalculates all measurement data via replot-> recalculate - optimize later if needed
        """
        self.machine.measurements = {}

        for mmt_child in self.mmt_group.children():
            mmt_name = mmt_child.name()
            artefact = dc.default_artefacts[mmt_child.child("artefact").value()]

            grp_loc = mmt_child.child("grp_location")
            vloc = [grand_kid.value() for grand_kid in grp_loc]
            grp_rot = mmt_child.child("grp_rotation")
            vrot = [grand_kid.value() for grand_kid in grp_rot]
            transform3d = gc.vec_to_transform3d(vloc, vrot).matrix()

            probe = self.machine.probes[mmt_child.child("probe").value()]
            mmt = dc.Measurement(
                title=mmt_child.title(),
                name=mmt_child.name(),
                artefact=artefact,
                transform_mat=transform3d,
                probe=probe,
                cmm_nominal=None,
                cmm_dev=None,
                mmt_nominal=None,
                mmt_dev=None,
            )
            self.machine.measurements[mmt_name] = mmt

        for dock in self.plot2d_docks:
            dock.update_measurement_list()
        if self.plot3d_dock:
            self.plot3d_dock.update_measurement_list()
        self.replot()

    def change_prb_title(self, param):
        """
        event handler for a change in probe name
        """
        param.parent().setOpts(title=param.value())
        try:
            self.update_measurements()
            # update probe titles in measurement lists
            for mmt_child in self.mmt_group.children():
                prb_limits = {
                    value.title: key for key, value in self.machine.probes.items()
                }
                mmt_child.child("probe").setLimits(prb_limits)
            # update probe titles in 3d dock list
            self.plot3d_dock.probe_box.setLimits(prb_limits)

        except AttributeError:
            # no mmt group yet
            pass

    def change_mmt_title(self, param):
        """
        event handler for a change in probe name
        """
        param.parent().setOpts(title=param.value())
        self.update_measurements()

    def add_startup_docks(self):
        """
        add the 3d plot dock
        """
        self.plot3d_dock = gc.Plot3dDock("3D Deformation", self.machine)
        self.dock_area.addDock(self.plot3d_dock)

    def add_new_plot2d_dock(self):
        """
        create a new plot dock
        type of plot etc is set from side bar on dock
        can have lots of these
        """
        new_plot_dock = gc.Plot2dDock("New Dock", self.machine)
        self.dock_area.addDock(new_plot_dock, position="bottom")
        self.plot2d_docks.append(new_plot_dock)
        new_plot_dock.replot()

    def recalculate(self):
        self.machine.recalculate()

    def replot(self):
        self.recalculate()
        if self.plot3d_dock:
            self.plot3d_dock.replot()
        for dock in self.plot2d_docks:
            dock.replot()

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
