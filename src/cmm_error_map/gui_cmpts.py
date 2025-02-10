# from dataclasses import dataclass, field
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Qt.QtGui as qtg
import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.dockarea.Dock import Dock, DockLabel
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt.QtCore import Qt as qtc

import cmm_error_map.data_cmpts as dc

logger = logging.getLogger(__name__)

# =========== THEME===============
# qdarktheme setup parameters
# https://pyqtdarktheme.readthedocs.io/en/stable/reference/qdarktheme.html
main_theme = dict(
    theme="dark",
    custom_colors={
        "primary": "#f07845",
        "list.alternateBackground": "#202124",
        "primary>button.hoverBackground": "#42444a",
    },
    corner_shape="sharp",
)

# css for individual widgets
tree_style = """
            ParameterControlledButton { background: #f07845; color: white;padding: 5px 10px 5px 10px;font-weight: bold;}
            ParameterControlledButton:hover { background: #42444a; border: 1px solid #f07845; padding:4px}
            """
add_btn_style = """
            QPushButton { background: #f07845; color: white;padding: 5px 10px 5px 10px;font-weight: bold;}
            QPushButton:hover { background: #42444a; border: 1px solid #f07845; padding:4px}
            """

slider_factors = {
    "Txx": 1e-6,
    "Txy": 1e-6,
    "Txz": 1e-6,
    "Tyx": 1e-6,
    "Tyy": 1e-6,
    "Tyz": 1e-6,
    "Tzx": 1e-6,
    "Tzy": 1e-6,
    "Tzz": 1e-6,
    "Rxx": 1e-8,
    "Rxy": 1e-8,
    "Rxz": 1e-8,
    "Ryx": 1e-8,
    "Ryy": 1e-8,
    "Ryz": 1e-8,
    "Rzx": 1e-8,
    "Rzy": 1e-8,
    "Rzz": 1e-8,
    "Wxy": 1e-8,
    "Wxz": 1e-8,
    "Wyz": 1e-8,
}

x = np.array([1, 2, 5, 7.5])
magnification_span = np.hstack((x * 100, x * 1000, x * 10000))

# used to assign controls to axis groups
axis_group = {
    "Txx": 0,
    "Txy": 0,
    "Txz": 0,
    "Tyx": 1,
    "Tyy": 1,
    "Tyz": 1,
    "Tzx": 2,
    "Tzy": 2,
    "Tzz": 2,
    "Rxx": 0,
    "Rxy": 0,
    "Rxz": 0,
    "Ryx": 1,
    "Ryy": 1,
    "Ryz": 1,
    "Rzx": 2,
    "Rzy": 2,
    "Rzz": 2,
    "Wxy": 3,
    "Wxz": 3,
    "Wyz": 3,
}


dock3d_control_grp = {
    "name": "dock3d_control_grp",
    "title": "▼",
    "type": "group",
    "children": [
        {
            "name": "probe_box",
            "title": "Probe for Box Deformation",
            "type": "list",
            "limits": [],
        },
        {
            "name": "mmts_to_plot",
            "title": "To Plot",
            "type": "checklist",
            "limits": [],
        },
        {
            "type": "slider",
            "name": "slider_mag",
            "title": "Magnification",
            "span": magnification_span,
            "value": 5000,
        },
    ],
}

grp_probe_lengths = {
    "title": "Probe Lengths",
    "name": "grp_probe_lengths",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "X",
            "title": "X/mm",
            "type": "float",
            "value": 0.0,
            "format": "{value:.3f}",
        },
        {
            "name": "Y",
            "title": "Y/mm",
            "type": "float",
            "value": 0.0,
            "format": "{value:.3f}",
        },
        {
            "name": "Z",
            "title": "Z/mm",
            "type": "float",
            "value": 0.0,
            "format": "{value:.3f}",
        },
    ],
}

probe_control_grp = {
    "title": "Probe 0",
    "name": "prb_control_grp0",
    "type": "group",
    "expanded": False,
    "context": ["Delete"],
    "children": [
        {
            "name": "prb_title",
            "title": "Probe Name",
            "type": "str",
            "value": "Probe 0",
        },
        grp_probe_lengths,
    ],
}

grp_location = {
    "title": "Location",
    "name": "grp_location",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "X",
            "title": "X/mm",
            "type": "float",
            "value": 0.000,
            "format": "{value:.3f}",
        },
        {
            "name": "Y",
            "title": "Y/mm",
            "type": "float",
            "value": 0.000,
            "format": "{value:.3f}",
        },
        {
            "name": "Z",
            "title": "Z/mm",
            "type": "float",
            "value": 0.000,
            "format": "{value:.3f}",
        },
        {
            "name": "centre",
            "title": "Centre on CMM",
            "type": "action",
            "value": 0,
        },
    ],
}
grp_rotation = {
    "title": "Rotation",
    "name": "grp_rotation",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "X",
            "title": "X/°",
            "type": "float",
            "value": 0.0,
            "format": "{value:.1f}",
        },
        {
            "name": "Y",
            "title": "Y/°",
            "type": "float",
            "value": 0.0,
            "format": "{value:.1f}",
        },
        {
            "name": "Z",
            "title": "Z/°",
            "type": "float",
            "value": 0.0,
            "format": "{value:.1f}",
        },
    ],
}

mmt_control_grp = {
    "name": "mmt_control_grp0",
    "title": "New Measurement",
    "type": "group",
    "context": ["Save to CSV", "Delete"],
    "children": [
        {
            "type": "str",
            "name": "mmt_title",
            "title": "Simulation Title",
            "value": "New Simulation",
        },
        {
            "name": "artefact",
            "title": "artefact type",
            "type": "list",
            "limits": [],
        },
        {
            "name": "probe",
            "title": "Probe",
            "type": "list",
            "limits": [],
        },
        grp_rotation,
        grp_location,
        {
            "name": "pen",
            "title": "Colour",
            "type": "pen",
            "expanded": False,
        },
    ],
}

default_pen = pg.mkPen({"color": "#bbb", "width": 1})
# parameter structures for 2d plot controls


dock2d_control_grp = {
    "title": "▼",
    "name": "plot_controls",
    "type": "group",
    "expanded": True,
    "children": [
        {
            "name": "plot_title",
            "title": "Plot Title",
            "type": "str",
            "value": "Plot 0",
        },
        {
            "name": "mmts_to_plot",
            "title": "To Plot",
            "type": "checklist",
            "limits": [],
        },
        {
            "type": "slider",
            "name": "slider_mag",
            "title": "Magnification",
            "span": magnification_span,
            "value": 5000,
        },
    ],
}

dock1d_control_grp = {
    "title": "▼",
    "name": "plot_controls",
    "type": "group",
    "expanded": True,
    "children": [
        {
            "name": "plot_title",
            "title": "Plot Title",
            "type": "str",
            "value": "Plot 0",
        },
        {
            "name": "mmts_to_plot",
            "title": "To Plot",
            "type": "checklist",
            "limits": [],
        },
    ],
}


def all_descendants(root, kids=None):
    """
    finds all child parameters recursively
    """
    if root is None:
        return
    if kids is None:
        kids = []
    if root.isType("group"):
        for child in root.children():
            all_descendants(child, kids)
    else:
        kids.append(root)
    return kids


def set_children_readonly(param, readonly):
    """
    sets all the child parameters of param to readonly status
    """
    with param.treeChangeBlocker():
        # only emit one signal for all changes
        kids = all_descendants(param)
        for kid in kids:
            kid.setOpts(readonly=readonly)


# MARK: 3D plots


def plot3d_axis(w):
    # axis items
    org = gl.GLScatterPlotItem(
        pos=(0, 0, 0), size=20, color=pg.mkColor("white"), pxMode=False
    )

    w.addItem(org)
    axis = gl.GLAxisItem()
    axis.setSize(x=1000, y=700, z=700)
    w.addItem(axis)

    xlabel = gl.GLTextItem(pos=(1000.0, 0.0, 0.0), text="X")
    w.addItem(xlabel)
    ylabel = gl.GLTextItem(pos=(0.0, 700.0, 0.0), text="Y")
    w.addItem(ylabel)
    zlabel = gl.GLTextItem(pos=(0.0, 0.0, 700.0), text="Z")
    w.addItem(zlabel)


def plot3d_box(w: gl.GLViewWidget, box: dc.BoxGrid, col="white") -> gl.GLLinePlotItem:
    """
    produces a 3D plot of the undeformed machine ready for updating with
    deformation and magnification via update_plot_model3d
    """
    plot3d_axis(w)
    # undeformed box
    fixed_box = gl.GLLinePlotItem(color=pg.mkColor("green"), mode="lines")
    update_plot3d_box(fixed_box, box, 1.0)
    w.addItem(fixed_box)

    # model deformed box
    gridlines = gl.GLLinePlotItem(color=pg.mkColor(col), mode="lines")
    update_plot3d_box(gridlines, box, 1.0)

    w.addItem(gridlines)
    w.setCameraPosition(distance=2000)

    return gridlines


def update_plot3d_box(gridlines: gl.GLLinePlotItem, box: dc.BoxGrid, mag):
    """
    update a plot produced by plot3d_box with a new set of params
    """

    ind = grid_line_index(box.npts)
    pts = box.grid_nominal[:, ind] + mag * box.grid_dev[:, ind]
    gridlines.setData(pos=pts.T)


def plot3d_plate(
    w: gl.GLViewWidget, mmt: dc.Measurement, col="white"
) -> list[type[gl.GLGraphicsItem]]:
    """
    produces a 3d plot of a ballplate ready for updating with location, rotation, deformation
    and magnification by update_plot_plate3d
    """
    # undeformed plate with no transform, zero length probe
    balls = gl.GLScatterPlotItem(color=pg.mkColor(col), size=20)
    lines = gl.GLLinePlotItem(mode="lines", color=pg.mkColor(col))

    update_plot3d_plate(
        balls,
        lines,
        mmt,
        magnification=1.0,
    )

    nx, ny = mmt.artefact.nballs
    xindex = nx - 1
    yindex = nx * (ny - 1)

    cols = [1, 1, 1, 0.75] * (nx * ny)  # white
    cols = np.array(cols).reshape(-1, 4)
    cols[0, :] = [1.0, 0.627, 0.157, 0.75]  # orange
    cols[xindex, :] = [1.0, 0.2, 0.322, 0.75]  # red
    cols[yindex, :] = [0.545, 0.863, 0, 0.75]  # green
    balls.setData(color=cols)

    w.addItem(balls)
    w.addItem(lines)

    return [balls, lines]


def update_plot3d_plate(
    balls: gl.GLScatterPlotItem,
    lines: gl.GLLinePlotItem,
    mmt: dc.Measurement,
    magnification: float,
):
    """
    takes the 3d plate position data mmt from mmt applies the magnifiacation
    and sets the data in balls and lines
    """
    xyz = mmt.cmm_nominal + magnification * mmt.cmm_dev
    balls.setData(pos=xyz.T)

    nx, ny = mmt.artefact.nballs
    ind = np.arange(nx * ny).reshape((ny, nx))
    indx = np.repeat(ind, 2, axis=1)[:, 1:-1].flatten()
    indy = np.repeat(ind, 2, axis=0)[1:-1, :].T.flatten()
    ind_lines = np.hstack((indx, indy))
    pos = xyz[:, ind_lines]
    lines.setData(pos=pos.T)


class Plot3dDock(Dock):
    """
    a pyqtgraph Dock containing a 3D plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    """

    def __init__(self, name: str, machine: dc.Machine):
        label1 = CustomDockLabel(name, closable=False)
        super(Plot3dDock, self).__init__(name, label=label1, closable=False)

        self.magnification = 5000
        self.machine = machine

        self.plot_data: dict[str, list[gl.GLLinePlotItem]] = {}
        self.box_lineplot: gl.GLLinePlotItem = None
        self.pens: dict[str, list[qtg.QPen]] = {}

        self.plot_widget = gl.GLViewWidget()
        self.add_control_tree()

        self.addWidget(self.plot_widget)
        self.dock_name = name

    def add_control_tree(self):
        """
        adds the controls that go in the side bar of each 3d plot
        """
        self.plot_controls = Parameter.create(**dock3d_control_grp)

        slider_mag = self.plot_controls.child("slider_mag")
        slider_mag.sigValueChanged.connect(self.change_magnification)

        self.mmts_to_plot = self.plot_controls.child("mmts_to_plot")
        # checklist limits need to be lists and display current title
        limits = [value.title for value in self.machine.measurements.values()]
        self.mmts_to_plot.setLimits(limits)

        self.mmts_to_plot.sigValueChanged.connect(self.update_mmts_plotted)

        self.probe_box = self.plot_controls.child("probe_box")
        probe_choices = {value.title: key for key, value in self.machine.probes.items()}
        self.probe_box.setLimits(probe_choices)
        self.probe_box.sigValueChanged.connect(self.replot)

        self.tree = ParameterTree(self.plot_widget, showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)
        self.tree.setStyleSheet(
            "background:transparent;border-width: 0px; border-style: solid"
        )
        self.tree.setFixedWidth(500)
        self.tree.setHorizontalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.move(0, 0)

    def update_mmts_plotted(self):
        self.update_pens()
        self.replot()

    def update_plot_plates(self):
        """
        plots an outline of the selected measurements in this 3d plot
        plot_data_2d.transform3d
        """
        # check for deletions from machine.measurements
        # need two steps as can't delete dict item during iteration
        to_delete = []
        for mmt_name in self.plot_data:
            if mmt_name not in self.machine.measurements:
                to_delete.append(mmt_name)

        for mmt_name in to_delete:
            # remove plotlines
            for plot in self.plot_data[mmt_name]:
                self.plot_widget.removeItem(plot)
            del self.plot_data[mmt_name]

        for mmt_name, mmt in self.machine.measurements.items():
            if mmt.cmm_nominal is None:
                # measurement has not been recalculated yet
                continue
            to_plot = mmt.title in self.mmts_to_plot.value()
            if not to_plot and mmt_name in self.plot_data:
                # remove plotlines
                for plot in self.plot_data[mmt_name]:
                    self.plot_widget.removeItem(plot)
                del self.plot_data[mmt_name]

            if to_plot:
                if mmt_name not in self.plot_data:
                    # need a new plot
                    self.plot_data[mmt_name] = plot3d_plate(self.plot_widget, mmt)
                    self.update_pens()

                # update plot
                balls, lines = self.plot_data[mmt_name]
                update_plot3d_plate(
                    balls,
                    lines,
                    mmt,
                    self.magnification,
                )

    def change_magnification(self, control):
        """
        event handler for a change in magnification
        """
        self.magnification = control.value()
        self.replot()

    def update_measurement_list(self):
        """
        update the displayed measurement list to match self.machine
        """
        limits = [value.title for value in self.machine.measurements.values()]
        self.mmts_to_plot.setLimits(limits)

    def update_machine(self, machine: dc.Machine):
        """
        called when the machine is changed
        """
        self.machine = machine
        self.plot_widget.clear()
        self.box_lineplot = None
        self.plot_data = {}

    def update_pens(self):
        if len(self.plot_data) > 0:
            for mmt_name, list_items in self.plot_data.items():
                balls, lines = list_items
                col = self.pens[mmt_name].color()

                lines.setData(color=col)

    def replot(self):
        """
        redraws the 3d box deformation plot from data in self.machine.measurements
        """
        box = self.machine.boxes[self.probe_box.value()]
        if self.box_lineplot is None:
            # haven't plotted the box deformation yet or machine changed
            self.box_lineplot = plot3d_box(self.plot_widget, box, col="blue")

        update_plot3d_box(self.box_lineplot, box, self.magnification)
        self.update_plot_plates()


# MARK: 2D Plots


def plot2d_plate(
    w: pg.PlotWidget,
    mmt: dc.Measurement,
) -> list[pg.PlotDataItem]:
    """
    pyqtgraph 2d plot of ballplate errors
    basic undeformed plot
    """
    # fixed grid
    col = "white"
    grid = pg.PlotDataItem(pen=pg.mkColor(col), connect="pairs")
    xy = mmt.mmt_nominal
    nx, ny = mmt.artefact.nballs
    ind = np.arange(nx * ny).reshape((ny, nx))
    indx = np.repeat(ind, 2, axis=1)[:, 1:-1].flatten()
    indy = np.repeat(ind, 2, axis=0)[1:-1, :].T.flatten()
    ind_lines = np.hstack((indx, indy))
    pos = xy[:2, ind_lines]
    grid.setData(pos.T)
    # rescale
    xy_min = 0.0
    x_max = xy[0, :].max()
    y_max = xy[1, :].max()
    w.getPlotItem().setRange(pg.QtCore.QRectF(xy_min, xy_min, x_max, y_max))

    # plate
    balls = pg.PlotDataItem(symbolBrush=pg.mkColor(col), size=20, symbol="o", pen=None)
    lines = pg.PlotDataItem(pen=pg.mkColor(col), connect="pairs")

    update_plot2d_plate(balls, lines, mmt, magnification=1.0)
    w.addItem(grid)
    w.addItem(balls)
    w.addItem(lines)

    return [balls, lines]


def update_plot2d_plate(
    balls: pg.PlotDataItem,
    lines: pg.PlotDataItem,
    mmt: dc.Measurement,
    magnification: float,
):
    xy = mmt.mmt_nominal + magnification * mmt.mmt_dev
    balls.setData(xy[:2, :].T)

    nx, ny = mmt.artefact.nballs
    ind = np.arange(nx * ny).reshape((ny, nx))
    indx = np.repeat(ind, 2, axis=1)[:, 1:-1].flatten()
    indy = np.repeat(ind, 2, axis=0)[1:-1, :].T.flatten()
    ind_lines = np.hstack((indx, indy))
    pos = xy[:2, ind_lines]
    lines.setData(pos.T)


def plot1d_bar(
    w: pg.PlotWidget, mmt: dc.Measurement, col="white"
) -> list[pg.PlotDataItem]:
    """
    pyqtgraph plot of ball bar errors against position
    basic undeformed plot
    """

    balls = pg.PlotDataItem(color=pg.mkColor(col), size=20, symbol="o", pen=None)
    lines = pg.PlotDataItem(color=pg.mkColor(col), connect="all")

    update_plot1d_bar(balls, lines, mmt)
    w.addItem(balls)
    w.addItem(lines)

    return [balls, lines]


def update_plot1d_bar(
    balls: pg.PlotDataItem,
    lines: pg.PlotDataItem,
    mmt: dc.Measurement,
):
    x = mmt.mmt_nominal[0, :]
    y = mmt.mmt_dev[0, :]
    xy = np.stack((x, y))

    balls.setData(xy.T)
    lines.setData(xy.T)


class PlotPlateDock(Dock):
    """
    a pyqtgraph Dock containing a plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    For 2D aretefacts such as a ball plate or hole plate
    """

    def __init__(self, name, machine):
        label1 = CustomDockLabel(name, closable=True)
        super(PlotPlateDock, self).__init__(name, label=label1, closable=True)

        self.magnification = 5000
        self.machine = machine
        self.dock_name = name
        self.pens = {}

        self.plot_data: dict[str, list[pg.PlotDataItem]] = {}
        self.plot_widget = pg.PlotWidget(name=name)
        self.plot_widget.setAspectLocked()
        self.plot_widget.getPlotItem().disableAutoRange()
        self.add_control_tree()

        self.addWidget(self.plot_widget)

    def add_control_tree(self):
        """
        adds the controls that go in the overlay at top left
        """
        self.plot_controls = Parameter.create(**dock2d_control_grp)
        plot_title = self.plot_controls.child("plot_title")
        plot_title.sigValueChanged.connect(self.change_plot_title)

        slider_mag = self.plot_controls.child("slider_mag")
        slider_mag.sigValueChanged.connect(self.change_magnification)

        self.mmts_to_plot = self.plot_controls.child("mmts_to_plot")
        self.update_measurement_list()
        self.mmts_to_plot.sigValueChanged.connect(self.update_display)

        self.tree = ParameterTree(self.plot_widget, showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)
        self.tree.setStyleSheet(
            "background:transparent;border-width: 0px; border-style: solid"
        )
        self.tree.setFixedWidth(500)
        self.tree.setHorizontalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.move(0, 0)

    def update_machine(self, machine: dc.Machine):
        self.machine = machine

    def update_display(self):
        self.update_pens()
        self.replot()

    def update_pens(self):
        logger.debug(list(self.pens.keys()))
        for mmt_name, list_items in self.plot_data.items():
            # debug
            if (mmt_name == "mmt_control_grp0") and (
                "mmt_control_grp0" not in self.pens.keys()
            ):
                # I'm about to throw an error
                # so lets add a breakpoint
                logger.debug(list(self.pens.keys()))

            balls, lines = list_items
            lines.setPen(self.pens[mmt_name])
            balls.setSymbolPen(self.pens[mmt_name])

    def update_measurement_list(self):
        """
        update the displayed measurement list to match self.machine
        """
        limits = []
        for mmt in self.machine.measurements.values():
            if mmt.artefact.nballs[1] > 1:
                limits.append(mmt.title)
        self.mmts_to_plot.setLimits(limits)

    def change_plot_title(self, param):
        """
        event handler for a change in plot title
        """
        param.parent().setOpts(title=param.value())
        # TODO add  title on plot

    def change_magnification(self, control):
        """
        event handler for a change in magnification
        """
        self.magnification = control.value()
        self.replot()

    def replot(self, control=None):
        """
        redraw existing plots and add new ones from data in self.machine.measurements
        """
        # check for deletions from machine.measurements
        # need two steps as can't delete dict item during iteration
        to_delete = []
        for mmt_name in self.plot_data:
            if mmt_name not in self.machine.measurements:
                to_delete.append(mmt_name)

        for mmt_name in to_delete:
            # remove plotlines
            for plot in self.plot_data[mmt_name]:
                self.plot_widget.removeItem(plot)
            del self.plot_data[mmt_name]

        for mmt_name, mmt in self.machine.measurements.items():
            if mmt.cmm_nominal is None:
                # measurement has not been recalculated yet
                continue
            to_plot = mmt.title in self.mmts_to_plot.value()
            if not to_plot and mmt_name in self.plot_data:
                # remove plotlines
                for plot in self.plot_data[mmt_name]:
                    self.plot_widget.removeItem(plot)
                del self.plot_data[mmt_name]

            if to_plot:
                if mmt_name not in self.plot_data:
                    # need a new plot
                    self.plot_data[mmt_name] = plot2d_plate(self.plot_widget, mmt)
                    self.update_pens()

                # update plot

                balls, lines = self.plot_data[mmt_name]
                update_plot2d_plate(
                    balls,
                    lines,
                    mmt,
                    self.magnification,
                )


class PlotBarDock(Dock):
    """
    a pyqtgraph Dock containing a plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    For 1D artefacts like a ball bar or step gauge
    """

    def __init__(self, name, machine):
        label1 = CustomDockLabel(name, closable=True)
        super(PlotBarDock, self).__init__(name, label=label1, closable=True)

        self.machine = machine
        self.dock_name = name

        self.plot_data: dict[str, list[pg.PlotDataItem]] = {}
        self.plot_widget = pg.PlotWidget(name=name)
        self.add_control_tree()

        self.addWidget(self.plot_widget)

    def add_control_tree(self):
        """
        adds the controls that go in the overlay at top left
        """
        self.plot_controls = Parameter.create(**dock1d_control_grp)
        plot_title = self.plot_controls.child("plot_title")
        plot_title.sigValueChanged.connect(self.change_plot_title)

        self.mmts_to_plot = self.plot_controls.child("mmts_to_plot")
        self.update_measurement_list()
        self.mmts_to_plot.sigValueChanged.connect(self.replot)

        self.tree = ParameterTree(self.plot_widget, showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)
        self.tree.setStyleSheet(
            "background:transparent;border-width: 0px; border-style: solid"
        )
        self.tree.setFixedWidth(500)
        self.tree.setHorizontalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.move(0, 0)

    def update_machine(self, machine: dc.Machine):
        self.machine = machine

    def update_measurement_list(self):
        """
        update the displayed measurement list to match self.machine
        """
        limits = []
        for mmt in self.machine.measurements.values():
            if mmt.artefact.nballs[1] == 1:
                limits.append(mmt.title)
        self.mmts_to_plot.setLimits(limits)

    def change_plot_title(self, param):
        """
        event handler for a change in plot title
        """
        param.parent().setOpts(title=param.value())
        # TODO add  title on plot

    def replot(self, control=None):
        """
        redraw existing plots and add new ones from data in self.machine.measurements
        """
        # check for deletions from machine.measurements
        # need two steps as can't delete dict item during iteration
        to_delete = []
        for mmt_name in self.plot_data:
            if mmt_name not in self.machine.measurements:
                to_delete.append(mmt_name)

        for mmt_name in to_delete:
            # remove plotlines
            for plot in self.plot_data[mmt_name]:
                self.plot_widget.removeItem(plot)
            del self.plot_data[mmt_name]

        for mmt_name, mmt in self.machine.measurements.items():
            if mmt.cmm_nominal is None:
                # measurement has not been recalculated yet
                continue

            to_plot = mmt.title in self.mmts_to_plot.value()
            if not to_plot and mmt_name in self.plot_data:
                # remove plotlines
                for plot in self.plot_data[mmt_name]:
                    self.plot_widget.removeItem(plot)
                del self.plot_data[mmt_name]

            if to_plot:
                if mmt_name not in self.plot_data:
                    # need a new plot
                    self.plot_data[mmt_name] = plot1d_bar(self.plot_widget, mmt)

                # update plot
                balls, lines = self.plot_data[mmt_name]

                update_plot1d_bar(
                    balls,
                    lines,
                    mmt,
                )


def grid_line_index(size=(5, 4, 4)):
    nx, ny, nz = size
    indicies = []
    indx = np.repeat(np.arange(nx), 2)[1:-1]
    indy = np.repeat(np.arange(ny), 2)[1:-1]
    indz = np.repeat(np.arange(nz), 2)[1:-1]

    # lines parallel to x-axis
    for k in range(nz):
        for j in range(ny):
            ind = [i + j * nx + k * nx * ny for i in indx]
            indicies.extend(ind)

    # lines parallel to y-axis
    for k in range(nz):
        for i in range(nx):
            ind = [i + j * nx + k * nx * ny for j in indy]
            indicies.extend(ind)

    # lines parallel to z-axis
    for j in range(ny):
        for i in range(nx):
            ind = [i + j * nx + k * nx * ny for k in indz]
            indicies.extend(ind)

    return indicies


class FileSaveTree(qtw.QTreeWidget):
    """
    display multiple file names for saving a  set of files
    file names are displayed as a tree
    the root folder can be edited (double click brings up file dialog) as can
    the proposed folder and file names.
    Only allows one folder deep for now
    """

    def __init__(
        self, root_folder: Path, folder_prefix: str, filenames: dict, parent=None
    ):
        super(FileSaveTree, self).__init__(parent)

        self.root_folder = root_folder
        self.filenames = filenames
        root_item = qtw.QTreeWidgetItem(self, [self.root_folder.as_posix()])
        folder = f"{folder_prefix}{dt.datetime.now().isoformat(sep='T')[:16]}"
        self.folder_item = qtw.QTreeWidgetItem(root_item, [folder])
        self.folder_item.setFlags(self.folder_item.flags() | qtc.ItemIsEditable)
        self.file_items = {}
        for key, fn in self.filenames.items():
            file_item = qtw.QTreeWidgetItem(self.folder_item, [fn])
            file_item.setFlags(file_item.flags() | qtc.ItemIsEditable)
            self.file_items[key] = file_item

        self.itemDoubleClicked.connect(self.edit_root_folder)
        self.expandAll()
        self.setItemsExpandable(False)
        self.setHeaderHidden(True)

    def edit_root_folder(self):
        parent_index = self.indexFromItem(self.currentItem()).parent().row()
        if parent_index == -1:
            folder = qtw.QFileDialog.getExistingDirectory()
            if not folder:
                return
            self.currentItem().setText(0, folder)

    def get_filenames(self):
        folder = self.root_folder / self.folder_item.text(0)
        for key, item in self.file_items.items():
            self.filenames[key] = folder / item.text(0)
        return self.filenames


class SaveSimulationDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__()
        layout = qtw.QVBoxLayout()
        self.setWindowTitle("Save Simulation to CSV files")
        text = (
            "The following files will be saved\n"
            "'snapshot.csv' \t- the minimum data to save for reimporting\n"
            "'fulldata.csv' \t- the simulation data in CMM and artefact coordinate systems\n"
            "'metadata.csv' \t- machine, probe, aretfact and model parmeters\n"
            "'readme.txt'   \t- the contents of the below text field\n"
            "\n"
            "The file and folder names can be edited by double clicking on them\n"
        )
        layout.addWidget(qtw.QLabel(text))

        home = Path.home() / "Desktop"
        self.files = FileSaveTree(
            root_folder=home,
            folder_prefix="simulation_",
            filenames={
                "snapshot": "snapshot.csv",
                "fulldata": "fulldata.csv",
                "metadata": "metadata.csv",
                "readme": "readme.txt",
            },
        )
        self.readme = qtw.QPlainTextEdit()
        self.readme.setPlaceholderText("...additional description of simulation...")

        btns = qtw.QDialogButtonBox.Save | qtw.QDialogButtonBox.Cancel
        buttonBox = qtw.QDialogButtonBox(btns)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.files)
        layout.addWidget(self.readme)
        layout.addWidget(buttonBox)

        self.setLayout(layout)

    def accept(self):
        self.readme_text = self.readme.toPlainText()
        self.filenames = self.files.get_filenames()
        super().accept()


class CustomDockLabel(DockLabel):
    def __init__(self, text, closable=True, fontSize="14px"):
        super(CustomDockLabel, self).__init__(text, closable, fontSize)

    def updateStyle(self):
        r = "1px"
        if self.dim:
            fg = "#aaa"
            bg = "#4f2f21"
            border = "#42444a"
        else:
            fg = "#fff"
            bg = "#f07845"
            border = "#42444a"

        if self.orientation == "vertical":
            self.vStyle = """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: 0px;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: %s;
                border-width: 0px;
                border-right: 1px solid %s;
                padding-top: 3px;
                padding-bottom: 3px;
                padding-left: 3px;
                padding-right: 3px;
                font-size: %s;
            }""" % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: %s;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-width: 0px;
                border-bottom: 1px solid %s;
                padding-top: 0px;
                padding-bottom: 0px;
                padding-left: 3px;
                padding-right: 3px;
                font-size: %s;

            }""" % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.hStyle)
