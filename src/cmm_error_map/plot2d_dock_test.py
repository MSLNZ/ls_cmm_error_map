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
        p0 = dc.Probe(title="P0", name="p0", length=qtg.QVector3D(0, 0, 0))
        p1 = dc.Probe(title="P1", name="p1", length=qtg.QVector3D(100, 100, -200))
        self.machine.probes = {p0.name: p0, p1.name: p1}

        m1 = dc.Measurement(
            title="m1",
            name="mmt_00",
            artefact=dc.default_artefacts["KOBA 0620"],
            transform3d=pg.Transform3D(),
            probe=p0,
            data2d=None,
        )
        m2 = dc.Measurement(
            title="m2",
            name="mmt_01",
            artefact=dc.default_artefacts["KOBA 0620"],
            transform3d=pg.Transform3D(),
            probe=p1,
            data2d=None,
        )

        self.machine.measurements = {m1.name: m1, m2.name: m2}
        self.machine.recalculate()

        self.plot2d_docks = []
        self.plot3d_docks = []

        self.setup_gui()
        self.sync_gui()
        self.make_docks()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.make_slider_controls()
        self.make_measurement_controls()
        self.make_prb_controls()

        self.control_group = Parameter.create(type="group", name="main_controls")
        self.control_group.addChild(self.prb_group)
        self.control_group.addChild(self.mmt_group)
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

    def make_docks(self):
        """
        add a Plot2dDock and a Plot3dDock
        """

        plot_dock = gc.Plot2dDock("Plot", self.machine)
        plot_dock.mmts_to_plot.setValue(["m1"])
        plot_dock.replot()

        self.dock_area.addDock(plot_dock)
        self.plot2d_docks.append(plot_dock)

        plot3d_dock = gc.Plot3dDock("3D Deformation", self.machine)
        plot3d_dock.mmts_to_plot.setValue(["m1"])
        plot3d_dock.replot()

        self.dock_area.addDock(plot3d_dock)
        self.plot3d_docks.append(plot3d_dock)

    def make_slider_controls(self) -> Parameter:
        # create sliders
        self.slider_group = Parameter.create(
            type="group", title="Linear Model", name="linear_model"
        )
        x_axis = self.slider_group.addChild(
            dict(type="group", name="X axis", expanded=False)
        )
        y_axis = self.slider_group.addChild(
            dict(type="group", name="Y axis", expanded=False)
        )
        z_axis = self.slider_group.addChild(
            dict(type="group", name="Z axis", expanded=False)
        )
        squareness = self.slider_group.addChild(
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

        self.slider_group.addChild(
            dict(type="action", name="btn_reset_all", title="Reset Model")
        )

        self.slider_group.sigTreeStateChanged.connect(self.update_model)

    def make_measurement_controls(self) -> Parameter:
        """
        create from self.machine
        """
        self.mmt_group = Parameter.create(
            type="group",
            title="Measurements",
            name="mmt_group",
            addText="Add Measurement",
        )
        self.mmt_group.sigAddNew.connect(self.add_new_mmt)
        self.mmt_group.sigTreeStateChanged.connect(self.mmt_control_change)

    def add_new_mmt(self, parent):
        """
        add a new measurement to self.machine
        """
        new_title = f"Measurement {len(parent.childs):02d}"
        new_name = f"mmt_{len(parent.childs):02d}"
        key0 = list(self.machine.probes.keys())[0]
        new_mmt = dc.Measurement(
            title=new_title,
            artefact=dc.default_artefacts["KOBA 0620"],
            transform3d=pg.Transform3D(),
            probe=self.machine.probes[key0],
            data2d=None,
        )
        new_mmt.recalculate(self.machine.model_params)
        self.machine.measurements[new_name] = new_mmt

    def mmt_control_change(self):
        """
        event handler for a change in any of the measurement controls
        recreates self.machine.measurements from gui controls
        """
        self.machine.measurements = {}

        for mmt_child in self.mmt_group.children():
            mmt_name = mmt_child.name()
            artefact = dc.default_artefacts[mmt_child.child("artefact").value()]

            grp_loc = mmt_child.child("grp_location")
            vloc = [grand_kid.value() for grand_kid in grp_loc]
            grp_rot = mmt_child.child("grp_rotation")
            vrot = [grand_kid.value() for grand_kid in grp_rot]
            transform3d = gc.vec_to_transform3d(vloc, vrot)

            probe = self.machine.probes[mmt_child.child("probe").value()]
            mmt = dc.Measurement(
                title=mmt_child.title(),
                name=mmt_name,
                artefact=artefact,
                transform3d=transform3d,
                probe=probe,
                data2d=None,
            )
            self.machine.measurements[mmt_name] = mmt

        self.machine.recalculate()

    def change_mmt_title(self, param):
        """
        event handler for a change in measurement title
        """
        pass

    def make_prb_controls(self) -> Parameter:
        """
        make control group for probes
        """
        self.prb_group = Parameter.create(
            type="group",
            title="Probes",
            name="probes_group",
            addText="Add Probe",
        )
        self.sync_gui()
        self.prb_group.sigAddNew.connect(self.add_new_prb)
        self.prb_group.sigTreeStateChanged.connect(self.prb_control_change)

    def add_new_prb(self, parent):
        """
        add a new probe to self.machine
        """
        new_title = f"Probe {len(parent.childs):02d}"
        new_name = f"prb_{len(parent.childs):02d}"
        new_prb = dc.Probe(title=new_title, length=qtg.QVector3D())
        self.machine.probes[new_name] = new_prb

    def prb_control_change(self):
        """
        event handler for a change in any of the probe controls
        recreates self.machine.measurements from gui controls
        """
        self.machine.probes = {}
        for probe_child in self.prb_group.children():
            probe_name = probe_child.name()
            grp_probe = probe_child.child("grp_probe_lengths")
            vprobe = [grand_kid.value() for grand_kid in grp_probe]
            probe_vec = qtg.QVector3D(*vprobe)
            probe = dc.Probe(
                title=probe_child.title(), name=probe_name, length=probe_vec
            )
            self.machine.probes[probe_name] = probe

        self.machine.recalculate()

    def change_prb_title(self, param):
        """
        event handler for a change in probe name
        """
        pass

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

    def replot(self):
        self.machine.recalculate()
        for dock in self.plot2d_docks:
            dock.replot()
        for dock in self.plot3d_docks:
            dock.replot()

    def sync_gui(self):
        """
        syncs gui from self.machine.measurements and self.machine.probes
        don't sync location, rotations as they're stored as a transform matrix - YET
        """
        with self.prb_group.treeChangeBlocker():
            prb_group_names = [child.name() for child in self.prb_group.children()]
            for prb_name, prb in self.machine.probes.items():
                if prb_name not in prb_group_names:
                    # new probe group
                    grp_params = gc.probe_control_grp.copy()
                    grp_params["name"] = prb_name
                    prb_child = self.prb_group.addChild(grp_params)

                prb_child = self.prb_group.child(prb_name)
                prb_child.setOpts(title=prb.title)
                prb_child.child("prb_title").setValue(prb.title)

        with self.mmt_group.treeChangeBlocker():
            mmt_group_names = [child.name() for child in self.mmt_group.children()]
            for mmt_name, mmt in self.machine.measurements.items():
                if mmt_name not in mmt_group_names:
                    # new measurement group
                    grp_params = gc.mmt_control_grp.copy()
                    grp_params["name"] = mmt_name
                    mmt_child = self.mmt_group.addChild(grp_params)

                mmt_child = self.mmt_group.child(mmt_name)
                mmt_child = self.mmt_group.child(mmt_name)
                mmt_child.setOpts(title=mmt.title)
                mmt_child.child("mmt_title").setValue(mmt.title)
                mmt_child.child("artefact").setValue(mmt.artefact.title)
                prb_child = mmt_child.child("probe")
                prb_child.setValue(mmt.probe.name)
                prb_choices = {
                    value.title: key for key, value in self.machine.probes.items()
                }
                prb_child.setLimits(prb_choices)
        # update measurement check lists in docks
        # TODO
        # update probe list in 3d dock
        # TODO


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
