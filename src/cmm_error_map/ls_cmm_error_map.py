"""
main gui for cmm error map app
0oOilL1I| 0123456789
"""

import datetime as dt
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.Qt.QtWidgets as qtw

# import pyqtgraph.Qt.QtGui as qtg
import qdarktheme
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt.QtCore import Qt as qtc

import cmm_error_map.config as cf
import cmm_error_map.data_cmpts as dc
import cmm_error_map.gui_cmpts as gc
from cmm_error_map import __version__

logging.basicConfig(
    # filename=cf.log_folder / "cmm_error_map.log",
    encoding="utf-8",
    filemode="w",
    format="%(asctime)s %(relativeCreated)8d %(levelname)-8s %(name)-30s %(funcName)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)

DEBUG = False


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.machine = dc.pmm_866
        self.cmm_models = cf.cmm_models
        self.nprbs = 0
        self.nmmts = 0
        self.pens = {}
        self.freeze_gui = False
        self.poly_txts = {}
        self.plot_docks = {}
        self.plot3d_dock = None
        self.setup_gui()
        self.add_startup_docks()

    def setup_gui(self):
        self.dock_area = DockArea()
        self.summary = qtw.QTextEdit()
        self.make_model_gui()
        self.make_probe_controls()
        self.make_measurement_controls()
        self.make_snapshot_controls()
        self.make_machine_controls()

        self.control_group = Parameter.create(type="group", name="Measurement Setup")

        self.control_group.addChild(self.machine_group)
        self.control_group.addChild(self.prb_group)
        self.control_group.addChild(self.mmt_group)
        self.control_group.addChild(self.snapshot_group)

        # other controls
        btn_plot2d = self.control_group.addChild(
            dict(type="action", name="btn_plot2d", title="Add Plate Plot Dock")
        )
        btn_plot2d.sigActivated.connect(self.add_new_plot_plate_dock)

        btn_plot1d = self.control_group.addChild(
            dict(type="action", name="btn_plot1d", title="Add Bar Plot Dock")
        )
        btn_plot1d.sigActivated.connect(self.add_new_plot_bar_dock)

        btn_save_state = self.control_group.addChild(
            dict(type="action", name="btn_save_state", title="Save Configuration")
        )
        btn_save_state.sigActivated.connect(self.save_state)

        btn_restore_state = self.control_group.addChild(
            dict(type="action", name="btn_restore_state", title="Restore Configuration")
        )
        btn_restore_state.sigActivated.connect(self.restore_state)

        # button for debug purposes
        if DEBUG:
            btn_debug = self.control_group.addChild(
                dict(type="action", name="btn_debug", title="DO NOT PUSH ME")
            )
            btn_debug.sigActivated.connect(self.btn_debug)

        self.control_tree = ParameterTree(showHeader=False)
        self.control_tree.setContentsMargins(0, 0, 0, 0)
        self.control_tree.header().setStretchLastSection(True)
        self.control_tree.header().setMinimumSectionSize(100)
        self.control_tree.setParameters(self.control_group, showTop=True)

        self.model_tree = ParameterTree(showHeader=False)
        self.model_tree.setContentsMargins(0, 0, 0, 0)
        self.model_tree.header().setStretchLastSection(True)
        self.model_tree.header().setMinimumSectionSize(100)
        self.model_tree.setParameters(self.slider_group, showTop=True)

        v_split = qtw.QSplitter(qtc.Vertical)
        v_split.addWidget(self.control_tree)
        v_split.addWidget(self.model_tree)
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
        self.extra_styling()

    def extra_styling(self):
        """
        extra style sheets as needed
        best done after gui created
        """
        self.control_tree.setStyleSheet(gc.tree_style)
        self.model_tree.setStyleSheet(gc.tree_style)
        for grp in [self.prb_group, self.mmt_group, self.snapshot_group]:
            add_btn = list(grp.items.keys())[0].addWidget
            add_btn.setStyleSheet(gc.add_btn_style)

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
        if self.freeze_gui:
            return
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
            self.machine.model_params[control_name] = [
                0.0,
                slider_value * slider_factor,
            ]

        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            dock.update_machine(self.machine)

        # this will call replot twice - optimize if needed
        self.update_probes()
        self.update_measurements()
        self.update_prb_lists()

    def make_model_gui(self):
        # create sliders
        self.slider_group = Parameter.create(
            type="group", title="Error Model", name="linear_model"
        )
        self.slider_group.addChild(
            dict(type="bool", title="Show Polynomials", name="poly_option", value=False)
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
        self.slider_axes = [x_axis, y_axis, z_axis, squareness]

        for key, axis in gc.axis_group.items():
            self.slider_axes[axis].addChild(
                dict(
                    type="slider",
                    name=key,
                    title=key,
                    limits=[-5.0, 5.0],
                    step=0.1,
                    value=0,
                    default=0,
                ),
            )

            poly_txt = self.slider_axes[axis].addChild(
                dict(
                    type="str",
                    name=key + "_poly",
                    title="    poly",
                    value="0.0, 1.0",
                    default="0.0, 1.0",
                ),
            )
            poly_txt.show(False)
            self.poly_txts[key] = poly_txt

        self.slider_group.addChild(
            dict(type="action", name="btn_reset_all", title="Reset Model")
        )

        self.slider_group.sigTreeStateChanged.connect(self.update_model)

    def parse_poly_str(self, text: str) -> np.array:
        """
        take gui poly field input and return array of coefficients
        """
        try:
            coeffs = [float(c) for c in text.split(",")]
            coeffs = np.array(coeffs)
        except ValueError:
            coeffs = None
        return coeffs

    def update_model(self, group, changes):
        """
        event callback for sliders
        """
        if self.freeze_gui:
            return
        control_name = changes[0][0].name()
        control_value = changes[0][2]
        if control_name[:3] in dc.model_parameters_zero.keys():
            slider_factor = gc.slider_factors[control_name[:3]]

            slider = changes[0][0].parent().child(control_name[:3])
            slider_value = slider.value()
            poly_txt = changes[0][0].parent().child(control_name[:3] + "_poly")
            coeffs = self.parse_poly_str(poly_txt.value())
            if coeffs is None:
                # can't interpret string in poly_txt
                poly_txt.setValue("0.0, 1.0")
                coeffs = np.array([0.0, 1.0])

            self.machine.model_params[control_name[:3]] = list(
                slider_value * slider_factor * coeffs
            )
        elif control_name == "poly_option":
            # show/hide controls
            self.freeze_gui = True

            for poly_txt in self.poly_txts.values():
                poly_txt.show(control_value)
            self.freeze_gui = False
        elif control_name == "btn_reset_all":
            self.machine.model_params = dc.model_parameters_zero.copy()
            # update sliders
            with self.slider_group.treeChangeBlocker():
                for axis_group in self.slider_group.children():
                    for child in axis_group.children():
                        "just reset sliders"
                        if child.name() in dc.model_parameters_zero.keys():
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
        new_title = f"Probe {self.nprbs}"
        grp_params = gc.probe_control_grp.copy()
        new_grp = parent.addChild(grp_params, autoIncrementName=True)
        new_grp.child("prb_title").sigValueChanged.connect(self.change_prb_title)
        new_grp.child("prb_title").setValue(new_title)
        new_grp.sigContextMenu.connect(self.delete_prb)
        self.nprbs += 1

    def mmt_menu(self, grp, change):
        if change == "Delete":
            self.delete_mmt(grp, change)
        elif change == "Save to CSV":
            mmt = self.machine.measurements[grp.name()]
            self.save_snapshot(mmt)

    def delete_mmt(self, grp, change):
        if change == "Delete":
            self.machine.measurements.pop(grp.name())
            self.recalculate()
            grp.remove()
            self.update_prb_lists()
            self.update_measurements()

    def delete_prb(self, grp, change):
        if change == "Delete":
            # check if probe is in use
            prb_in_use = False
            for mmt in self.machine.measurements.values():
                if mmt.probe.name == grp.name():
                    prb_in_use = True
            if self.plot3d_dock.probe_box.value() == grp.name():
                prb_in_use = True
            if prb_in_use:
                qtw.QMessageBox.warning(
                    self, "Warning", "Can not delete this probe\nas it is in use"
                )
                return
            self.machine.probes.pop(grp.name())
            grp.remove()
            self.update_prb_lists()
            self.update_measurements()

    def update_probes(self):
        """
        recreates self.machine.probes from gui controls in self.probes_group
        """
        if self.freeze_gui:
            return

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
            title="Simulations",
            name="mmt_group",
            addText="Add Simulation",
        )
        self.add_new_mmt_group(self.mmt_group)
        self.mmt_group.sigAddNew.connect(self.add_new_mmt_group)
        self.mmt_group.sigTreeStateChanged.connect(self.update_measurements)

    def add_new_mmt_group(self, parent):
        """
        add the controls for a new artefact measurement to the side bar
        """
        new_title = f"Simulation {self.nmmts}"
        grp_params = gc.mmt_control_grp.copy()

        with self.mmt_group.treeChangeBlocker():
            new_grp = parent.addChild(grp_params, autoIncrementName=True)
            new_grp.setOpts(title=new_title)
            new_grp.child("mmt_title").setValue(new_title)
            new_grp.child("artefact").setLimits(list(cf.artefact_models.keys()))
            new_grp.child("artefact").setValue(list(cf.artefact_models.keys())[0])
            prb_limits = {
                value.title: key for key, value in self.machine.probes.items()
            }
            new_grp.child("probe").setLimits(prb_limits)
            new_grp.child("probe").setValue(list(prb_limits.values())[0])

            new_grp.child("mmt_title").sigValueChanged.connect(self.change_mmt_title)
            new_grp.sigContextMenu.connect(self.mmt_menu)

            new_grp.child("grp_location", "centre").sigActivated.connect(
                self.centre_on_cmm
            )

        self.update_measurements()
        self.nmmts += 1

    def update_measurements(self, changes=None, info=None):
        """
        recreates self.machine.measurements from gui controls in self.mmt_group
        recalculates all measurement data via replot-> recalculate - optimize later if needed
        """

        # keep the snapshots
        if self.freeze_gui:
            return
        self.machine.measurements = {
            k: v for k, v in self.machine.measurements.items() if v.fixed
        }
        # recreate the simulations from gui
        for mmt_child in self.mmt_group.children():
            mmt_name = mmt_child.name()
            artefact = cf.artefact_models[mmt_child.child("artefact").value()]

            grp_loc = mmt_child.child("grp_location")
            vloc = [grand_kid.value() for grand_kid in grp_loc][:-1]
            grp_rot = mmt_child.child("grp_rotation")
            vrot = [grand_kid.value() for grand_kid in grp_rot]

            probe = self.machine.probes[mmt_child.child("probe").value()]

            mat = dc.matrix_from_vectors(vloc, vrot)
            mmt = dc.Measurement(
                title=mmt_child.title(),
                name=mmt_child.name(),
                artefact=artefact,
                transform_mat=mat,
                probe=probe,
                cmm_nominal=None,
                cmm_dev=None,
                mmt_nominal=None,
                mmt_dev=None,
            )
            self.machine.measurements[mmt_name] = mmt
            self.pens[mmt_name] = mmt_child.child("pen").pen

        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            dock.pens = self.pens
            dock.update_pens()
            dock.update_measurement_list()
        self.set_mmt_colors()

        self.replot()

    def set_mmt_colors(self):
        for mmt_child in self.mmt_group.children():
            mmt_name = mmt_child.name()
            pen = self.pens[mmt_name]
            try:
                pitem = list(mmt_child.items.keys())[0]
            except IndexError:
                return
            pitem.setForeground(0, pen.color())

    def centre_on_cmm(self, item):
        self.update_measurements()
        mmt_name = item.parent().parent().name()
        mmt = self.machine.measurements[mmt_name]
        grp_loc = self.mmt_group.child(mmt_name, "grp_location")
        old_vloc = [grand_kid.value() for grand_kid in grp_loc][:-1]
        cmm_centre = np.array(self.machine.cmm_model.size) / 2.0

        mmt_centre = (
            (np.array(mmt.artefact.nballs) - 1) * mmt.artefact.ball_spacing / 2.0
        )
        mmt_centre = np.append(mmt_centre, [0.0, 1.0]).T
        mmt_centre = mmt.transform_mat @ mmt_centre
        new_vloc = list(cmm_centre - mmt_centre[:-1] + old_vloc) + [0.0]

        [kid.setValue(x) for x, kid in zip(new_vloc, grp_loc)]

    def save_snapshot(self, mmt: dc.Measurement):
        dialog = gc.SaveSimulationDialog()
        if dialog.exec() == qtw.QDialog.Accepted:
            now = dt.datetime.now().isoformat(sep=" ")
            fns = dialog.filenames
            # create folder
            fns["snapshot"].parent.mkdir(parents=True, exist_ok=True)
            dc.mmt_snapshot_to_csv(fns["snapshot"], mmt, now)
            dc.mmt_full_data_to_csv(fns["fulldata"], mmt, now)
            dc.mmt_metadata_to_csv(fns["metadata"], mmt, self.machine, now)
            # readme
            pre_text = f"Created at: {now}\n"
            pre_text += f"CMM Error Map Software Version: {__version__}\n"
            exe_fn = Path(sys.executable)
            if exe_fn.name != "python.exe":
                # run from exe created with pyinstaller etc.
                pre_text += f"exe file: {exe_fn.resolve()} \n"
                pre_text += f"{dt.datetime.fromtimestamp(exe_fn.stat().st_ctime)}\n"

            with open(fns["readme"], "w") as fp:
                fp.write(pre_text)
                fp.write("\n\n")
                fp.write(dialog.readme_text)

    def make_snapshot_controls(self) -> Parameter:
        self.snapshot_group = Parameter.create(
            type="group",
            title="Snapshots",
            name="snapshot_group",
            addText="Load from CSV",
        )
        self.snapshot_group.sigAddNew.connect(self.load_snapshot)

    def load_snapshot(self):
        filename, _ = qtw.QFileDialog.getOpenFileName(self, filter="CSV Files (*.csv)")
        if not filename:
            return

        mmt = dc.mmt_from_snapshot_csv(Path(filename))
        mmt_name = f"snapshot_grp{len(self.snapshot_group.children())}"
        self.add_new_snapshot_group(mmt)
        self.machine.measurements[mmt_name] = mmt
        self.nmmts += 1

        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            dock.update_measurement_list()

        self.replot()

    def add_new_snapshot_group(self, mmt):
        # a snapshot is just a immutable measurement
        grp_params = gc.mmt_control_grp.copy()
        grp_params["name"] = "snapshot_grp0"
        grp_params["context"] = ["Delete"]
        new_grp = self.snapshot_group.addChild(grp_params, autoIncrementName=True)
        gc.set_children_readonly(new_grp, True)

        new_grp.setOpts(title=mmt.title)
        new_grp.child("mmt_title").setValue(mmt.title)
        new_grp.child("mmt_title").setOpts(title="Snapshot Title")
        new_grp.child("mmt_title").setReadonly(False)
        new_grp.child("mmt_title").sigValueChanged.connect(self.change_snapshot_title)
        new_grp.sigContextMenu.connect(self.mmt_menu)

        new_grp.child("artefact").setLimits([mmt.artefact.title])
        new_grp.child("artefact").setValue(mmt.artefact.title)

        new_grp.child("probe").setLimits(["NA"])
        new_grp.child("probe").setValue("NA")

        location, rotation = dc.matrix_to_vectors(mmt.transform_mat)
        for i, axis in enumerate(["X", "Y", "Z"]):
            new_grp.child("grp_location", axis).setValue(location[i])
            new_grp.child("grp_rotation", axis).setValue(rotation[i])

    def change_snapshot_title(self, param):
        param.parent().setOpts(title=param.value())
        self.machine.measurements[param.parent().name()].title = param.value()
        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            dock.update_measurement_list()

    def change_prb_title(self, param):
        """
        event handler for a change in probe name
        """
        param.parent().setOpts(title=param.value())
        self.update_prb_lists()

    def update_prb_lists(self):
        if self.freeze_gui:
            return
        try:
            self.update_measurements()
            # update probe titles in measurement lists
            for mmt_child in self.mmt_group.children():
                prb_p = mmt_child.child("probe")
                prb_v = prb_p.value()
                prb_limits = {
                    value.title: key for key, value in self.machine.probes.items()
                }
                mmt_child.child("probe").setLimits(prb_limits)
                # do emit rather than setValue as this makes sure lists update.
                prb_p.sigValueChanged.emit(prb_p, prb_v)

            # update probe titles in 3d dock list
            if len(self.mmt_group.children()) > 0:
                prb_p = self.plot3d_dock.probe_box
                prb_v = prb_p.value()
                prb_p.setLimits(prb_limits)
                prb_p.sigValueChanged.emit(prb_p, prb_v)
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
        self.update_measurements()

    def add_new_plot_plate_dock(self, _parameter, name=None):
        """
        create a new plot dock for a ball plate
        can have lots of these
        each dock can display multiple measurements
        """
        _containers, self.plot_docks = self.dock_area.findAll()
        if name is None:
            name = f"plate{len(self.plot_docks) - 1}"

        new_plot_dock = gc.PlotPlateDock(name, self.machine)

        self.dock_area.addDock(new_plot_dock, position="bottom")
        _containers, self.plot_docks = self.dock_area.findAll()
        new_plot_dock.replot()
        new_plot_dock.pens = self.pens
        new_plot_dock.update_pens()

    def add_new_plot_bar_dock(self, _parameter, name=None):
        """
        create a new plot dock for a ball bar
        can have lots of these
        each dock can display multiple measurements
        """
        _containers, self.plot_docks = self.dock_area.findAll()
        if name is None:
            name = f"bar{len(self.plot_docks)}"
        new_plot_dock = gc.PlotBarDock(name, self.machine)

        self.dock_area.addDock(new_plot_dock, position="bottom")
        _containers, self.plot_docks = self.dock_area.findAll()
        new_plot_dock.replot()

    def recalculate(self):
        self.machine.recalculate()

    def replot(self):
        self.recalculate()
        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            dock.replot()

        # if self.plot3d_dock:
        #     self.plot3d_dock.replot()
        # for dock in self.plot_mmt_docks:
        #     dock.replot()

    def save_state(self):
        filename, _ = qtw.QFileDialog.getSaveFileName(
            self, "Save File", dir="config.pkl", filter="pickle files (*.pkl)"
        )

        if not filename:
            return
        if Path(filename).suffix == "":
            filename = filename + ".pkl"
        state_dict = {}
        # state_dict["main_state"] = self.control_group.saveState(filter="user")
        state_dict["main_state"] = self.control_group.saveState()
        state_dict["counts"] = {
            "probes": len(self.prb_group.children()),
            "simulations": len(self.mmt_group.children()),
            "snapshots": len(self.snapshot_group.children()),
        }
        state_dict["dock_area"] = self.dock_area.saveState()
        state_dict["docks"] = {}

        _containers, self.plot_docks = self.dock_area.findAll()
        for dock in self.plot_docks.values():
            state_dict["docks"][dock.dock_name] = dock.plot_controls.saveState()
            # state_dict["docks"][dock.dock_name] = dock.plot_controls.saveState(
            #     filter="user"
            # )
        with open(filename, "wb") as fp:
            pickle.dump(state_dict, fp)

    def restore_state(self):
        self.freeze_gui = True
        filename, _ = qtw.QFileDialog.getOpenFileName(
            self, filter="config files (*.pkl)"
        )
        if not filename:
            return
        self.restore_filename(filename)

    def restore_filename(self, filename):
        with open(filename, "rb") as fp:
            state_dict = pickle.load(fp)

        # main control panel
        # remove existing children
        self.prb_group.clearChildren()
        self.mmt_group.clearChildren()
        self.snapshot_group.clearChildren()

        self.nprbs = 0
        self.nmmts = 0
        self.pens = {}

        # add the right number back
        for i in range(state_dict["counts"]["probes"]):
            self.add_new_probe_group(self.prb_group)
        for i in range(state_dict["counts"]["simulations"]):
            self.add_new_mmt_group(self.mmt_group)
        for i in range(state_dict["counts"]["snapshots"]):
            self.add_new_snapshot_group(self.snapshot_group)

        # restore state of main control panel
        self.control_group.restoreState(state_dict["main_state"])

        # remove any existing docks

        for dock in self.plot_docks.values():
            if dock.dock_name != "3D Deformation":
                dock.close()
                del dock

        _containers, self.plot_docks = self.dock_area.findAll()
        for dock_name, dock_state in state_dict["docks"].items():
            if dock_name[0] == "p":
                self.add_new_plot_plate_dock(None, name=dock_name)
                self.plot_docks[dock_name].plot_controls.restoreState(dock_state)

            elif dock_name[0] == "b":
                self.add_new_plot_bar_dock(name=dock_name)
                self.plot_docks[dock_name].plot_controls.restoreState(dock_state)
            elif dock_name[0] == "3":
                self.plot_docks[dock_name].plot_controls.restoreState(dock_state)
            else:
                raise ValueError("Unknown dock type")

        self.dock_area.restoreState(state_dict["dock_area"])
        self.freeze_gui = False
        self.update_machine()

    def btn_debug(self):
        """
        print useful stuff here
        or add to  summary etc.
        """
        logging.debug("before restore")
        filename = cf.config_folder / "gui_configs" / "Legex-3-axis-6-prbs.pkl"
        self.restore_filename(filename)
        logging.debug("after restore")


def main():
    _app = pg.mkQApp("CMM Error Map App")

    qdarktheme.setup_theme(**gc.main_theme)

    w = MainWindow()
    logging.debug("MainWindow created")
    w.showMaximized()
    w.show()
    logging.debug("showing")

    pg.exec()


if __name__ == "__main__":
    main()
