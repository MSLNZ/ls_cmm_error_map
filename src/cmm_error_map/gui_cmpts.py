# from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.parametertree import Parameter, ParameterTree

from pyqtgraph.Qt.QtCore import Qt as qtc
# import pyqtgraph.Qt.QtGui as qtg

from PySide6.QtCore import Signal

import cmm_error_map.design_matrix_linear_fixed as design
import cmm_error_map.data_cmpts as dc

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

# MARK: 3D plots


def plot_model3d(w: gl.GLViewWidget, col="white") -> list[gl.GLLinePlotItem]:
    """
    produces a 3D magniifed plot of the undeformed machine ready for updating with
    deformation via update_plot_model3d
    """
    params = [0.0] * 21
    XYZ, eXYZ = design.machine_deformation(params, 0.0, 0.0, 0.0)
    pXYZ_3D = XYZ + eXYZ

    w.setCameraPosition(distance=2000)

    nx, ny, nz = XYZ.shape[:3]

    plot_lines = []

    # lines parallel to x axis
    for j in range(ny):
        for k in range(nz):
            pts = pXYZ_3D[:, j, k, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            plot_lines.append(plt)

    # lines parallel to y axis
    for i in range(nx):
        for k in range(nz):
            pts = pXYZ_3D[i, :, k, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            plot_lines.append(plt)

    # lines parallel to z axis
    for i in range(nx):
        for j in range(ny):
            pts = pXYZ_3D[i, j, :, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            plot_lines.append(plt)

    # axis
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

    return plot_lines


def update_plot_model3d(
    plot_lines: list[gl.GLLinePlotItem], params: dict, xt, yt, zt, mag
):
    """
    update a plot produced by plot_model3d with a new set of params
    """
    pars = list(params.values())

    XYZ, eXYZ = design.machine_deformation(pars, xt, yt, zt)
    pXYZ_3D = XYZ + mag * eXYZ

    nx, ny, nz = XYZ.shape[:3]

    # lines parallel to x axis
    pn = 0
    for j in range(ny):
        for k in range(nz):
            pts = pXYZ_3D[:, j, k, :]
            plot_lines[pn].setData(pos=pts)
            pn += 1

    # lines parallel to y axis
    for i in range(nx):
        for k in range(nz):
            pts = pXYZ_3D[i, :, k, :]
            plot_lines[pn].setData(pos=pts)
            pn += 1

    # lines parallel to z axis
    for i in range(nx):
        for j in range(ny):
            pts = pXYZ_3D[i, j, :, :]
            plot_lines[pn].setData(pos=pts)
            pn += 1


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
        {"name": "X", "type": "float", "value": 0.0},
        {"name": "Y", "type": "float", "value": 0.0},
        {"name": "Z", "type": "float", "value": 0.0},
    ],
}

probe_control_grp = {
    "title": "Probe 0",
    "name": "prb_control_grp0",
    "type": "group",
    "expanded": False,
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
        {"name": "X", "type": "float", "value": 0.0},
        {"name": "Y", "type": "float", "value": 0.0},
        {"name": "Z", "type": "float", "value": 0.0},
    ],
}
grp_rotation = {
    "title": "Rotation",
    "name": "grp_rotation",
    "type": "group",
    "expanded": False,
    "children": [
        {"name": "X", "type": "float", "value": 0.0, "suffix": "°"},
        {"name": "Y", "type": "float", "value": 0.0, "suffix": "°"},
        {"name": "Z", "type": "float", "value": 0.0, "suffix": "°"},
    ],
}

mmt_control_grp = {
    "name": "mmt_control_grp0",
    "title": "New Measurement",
    "type": "group",
    "children": [
        {
            "type": "str",
            "name": "mmt_title",
            "title": "Measurement Title",
            "value": "New Measurement",
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
        grp_location,
        grp_rotation,
    ],
}


class Plot3dDock(Dock):
    """
    a pyqtgraph Dock containing a 3D plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    """

    def __init__(self, name: str, machine: dc.Machine):
        super(Plot3dDock, self).__init__(name)

        self.magnification = 5000
        self.machine = machine

        self.plot_data: dict[str, list[gl.GLLinePlotItem]] = {}
        self.box_lineplots: list[gl.GLLinePlotItem] = []

        self.plot_widget = gl.GLViewWidget()
        self.add_control_tree()

        # undeformed
        plot_model3d(self.plot_widget, col="green")

        self.addWidget(self.plot_widget)

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

        self.mmts_to_plot.sigValueChanged.connect(self.replot)

        self.probe_box = self.plot_controls.child("probe_box")
        probe_choices = {value.title: key for key, value in self.machine.probes.items()}
        self.probe_box.setLimits(probe_choices)
        self.probe_box.sigValueChanged.connect(self.replot)

        self.tree = ParameterTree(self.plot_widget, showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)
        self.tree.setStyleSheet(
            "background:transparent;" "border-width: 0px; border-style: solid"
        )
        self.tree.setFixedWidth(500)
        self.tree.setHorizontalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.move(0, 0)

    # def plot_ball_plate(self, artefact: dict, plot_data_2d: PlotData2d):
    #     """
    #     plots an outline of the given artefact at the transform defined by
    #     plot_data_2d.transform3d
    #     """
    #     new_artefact = ArtefactData3d()
    #     new_artefact.title = artefact["title"]
    #     new_artefact.ball_spacing = artefact["ball_spacing"]
    #     new_artefact.nballs = artefact["nballs"]

    #     xballs = artefact["nballs"][0]
    #     yballs = artefact["nballs"][0]
    #     ball_count = xballs * yballs
    #     ballnumber = np.arange(ball_count)
    #     ballspacing = artefact["ball_spacing"]
    #     sizex = (xballs - 1) * ballspacing
    #     sizey = (yballs - 1) * ballspacing

    #     xp = (ballnumber % xballs) * ballspacing
    #     yp = (ballnumber // yballs) * ballspacing
    #     xyz1 = np.stack((xp, yp, np.zeros_like(xp), np.ones_like(xp)))
    #     pts = plot_data_2d.transform3d.matrix() @ xyz1
    #     col = "red"
    #     new_artefact.points = gl.GLScatterPlotItem(pos=pts.T, color=pg.mkColor(col))

    #     grid = gl.GLGridItem(color=pg.mkColor("red"))
    #     grid.setSize(sizex, sizey, 0)
    #     grid.setSpacing(ballspacing, ballspacing, 0)
    #     grid.translate(sizex / 2, sizey / 2, 0)

    #     new_artefact.grid = grid
    #     self.artefacts[new_artefact.title] = new_artefact

    #     self.plot_widget.addItem(new_artefact.points)
    #     self.plot_widget.addItem(grid)

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

    def replot(self):
        """
        redraws the 3d box deformation plot from data in self.machine.measurements
        """
        probe_box_name = self.probe_box.value()
        vprobe = self.machine.probes[probe_box_name].length
        xt, yt, zt = vprobe.x(), vprobe.y(), vprobe.z()
        if len(self.box_lineplots) == 0:
            # haven't plotted the box deformation yet
            self.box_lineplots = plot_model3d(self.plot_widget, col="blue")

        update_plot_model3d(
            self.box_lineplots,
            self.machine.model_params,
            xt,
            yt,
            zt,
            self.magnification,
        )


# MARK: 2D Plots

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


# TODO these need to be parameters in gui or config
ballspacing = 133.0

U95 = 1.2


def single_grid_plot_data(
    dxy,
    mag,
    ballspacing=133.0,
    nballs=(5, 5),
    lines=True,
    circles=True,
):
    """
    drawing red crosses on out of tolerance point commented out
    until I figure how to update them (number of red crosses can change)
    dxy shape(25,2) or shape (20,2) single set of data to plot on current figure
    in order of ballnumber
    """
    ballnumber = np.arange(dxy.shape[0])
    xplaten = (ballnumber) % nballs[0]
    yplaten = (ballnumber) // nballs[1]

    xplot = mag * dxy[:, 0] + xplaten * ballspacing
    yplot = mag * dxy[:, 1] + yplaten * ballspacing

    data = []
    data.append((xplot, yplot))

    if lines:
        for i in range(0, 5):
            data.append((xplot[xplaten[:] == i], yplot[xplaten[:] == i]))
            data.append((xplot[yplaten[:] == i], yplot[yplaten[:] == i]))

    # if circles:
    #     # find points outside circles and mark with cross
    #     ballnumber = np.arange(dxy.shape[0])
    #     xplaten = (ballnumber) % 5
    #     yplaten = (ballnumber) // 5
    #     xcirc = xplaten * ballspacing
    #     ycirc = yplaten * ballspacing
    #     rcirc = mag * (U95 + ((xcirc**2 + ycirc**2) ** 0.5) / 400.0) * 1e-3
    #     err = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2) ** 0.5
    #     xout = xplot[err > rcirc / mag]
    #     yout = yplot[err > rcirc / mag]
    #     # data.append((xout, yout))
    return data


def plot_ballplate(
    plotw: pg.PlotWidget,
    ballspacing=133.0,
    nballs=(5, 5),
    lines=True,
    circles=True,
) -> list[pg.PlotDataItem]:
    """
    pyqtgraph 2d plot of ballplate errors
    basic undeformed plot
    """

    params = [0.0] * 21
    # plate transforamtion and position are arbitary here with params all zero
    RP = np.array(
        [
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 100.0],
            [0.0, 0.0, 1.0, 100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    xt, yt, zt = 0.0, 0.0, 0.0
    lineplots = []

    mag = 1
    dxy = design.modelled_mmts_XYZ(
        RP, xt, yt, zt, params, ballspacing=ballspacing, nballs=nballs
    )
    data = single_grid_plot_data(dxy, mag)
    di = 0
    p1 = plotw.plot(
        x=data[di][0],
        y=data[di][1],
        pen=None,
        symbol="o",
    )
    lineplots.append(p1)
    di += 1

    if lines:
        for i in range(0, 5):
            p2 = plotw.plot(x=data[di][0], y=data[di][1])
            di += 1
            lineplots.append(p2)
            p3 = plotw.plot(x=data[di][0], y=data[di][1])
            di += 1
            lineplots.append(p3)

    if circles:
        # draw uncertainty circles at nominal positions
        ballnumber = np.arange(data[0][0].shape[0])
        xplaten = (ballnumber) % 5
        yplaten = (ballnumber) // 5
        xcirc = xplaten * ballspacing
        ycirc = yplaten * ballspacing
        # TODO make U95 a function that can be configed
        # U95 = 1.2 + L/400 is hardcoded here
        rcirc = mag * (U95 + ((xcirc**2 + ycirc**2) ** 0.5) / 400.0) * 1e-3

        p4 = plotw.plot(
            x=xcirc, y=ycirc, pen=None, symbol="o", symbolSize=rcirc, pxMode=False
        )
        lineplots.append(p4)

        # # points outside circles are marked with cross
        # p5 = plotw.plot(
        #     x=data[di][0],
        #     y=data[di][1],
        #     symbol="x",
        #     symbolBrush="red",
        #     symbolSize=10,
        #     pen=None,
        # )
        # lineplots.append(p5)
        plotw.setAspectLocked()
        grid = pg.GridItem()
        grid.setTickSpacing(x=[ballspacing], y=[ballspacing])
        plotw.addItem(grid)

    return lineplots


class Plot2dDock(Dock):
    """
    a pyqtgraph Dock containing a plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    """

    update_gui = Signal((str,))

    def __init__(self, name, machine):
        super(Plot2dDock, self).__init__(name)

        self.magnification = 5000
        self.machine = machine

        self.plot_data: dict[str, list[pg.PlotDataItem]] = {}
        self.plot_widget = pg.PlotWidget(name=name)
        self.add_control_tree()

        self.addWidget(self.plot_widget)

    def request_update_gui(self):
        self.update_gui.emit(self.name)

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
        # checklist limits need to be lists and display current title
        limits = [value.title for value in self.machine.measurements.values()]
        self.mmts_to_plot.setLimits(limits)
        self.mmts_to_plot.sigValueChanged.connect(self.replot)

        self.tree = ParameterTree(self.plot_widget, showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)
        self.tree.setStyleSheet(
            "background:transparent;" "border-width: 0px; border-style: solid"
        )
        self.tree.setFixedWidth(500)
        self.tree.setHorizontalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollBarPolicy(qtc.ScrollBarAlwaysOff)
        self.tree.move(0, 0)

    def update_measurement_list(self):
        """
        update the displayed measurement list to match self.machine
        """
        limits = [value.title for value in self.machine.measurements.values()]
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

        for mmt_name, mmt in self.machine.measurements.items():
            to_plot = mmt.title in self.mmts_to_plot.value()
            if not to_plot and mmt_name in self.plot_data:
                # remove plotlines
                for plot in self.plot_data[mmt_name]:
                    self.plot_widget.removeItem(plot)
                del self.plot_data[mmt_name]

            if to_plot:
                if mmt_name not in self.plot_data:
                    # need a new plot
                    self.plot_data[mmt_name] = plot_ballplate(self.plot_widget)

                # update plot
                # convert measurement data to data needed to update PlotDataItems
                mmt = self.machine.measurements[mmt_name]
                dxy = mmt.data
                ballspacing = mmt.artefact.ball_spacing
                nballs = mmt.artefact.nballs
                data = single_grid_plot_data(
                    dxy,
                    self.magnification,
                    ballspacing=ballspacing,
                    nballs=nballs,
                )
                # update PlotDataItems with new data
                for datum, plot in zip(data, self.plot_data[mmt_name]):
                    plot.setData(x=datum[0], y=datum[1])


def vec_to_transform3d(vloc, vrot) -> pg.Transform3D:
    """
    takes the vectors from the gui elements (rot in degrees) and
    returns the corresponding Transform3D object
    sets the (euler) angles in order xyz to match the
    Blender mathutils Euler.to_matrix method
    """
    mat = pg.Transform3D()
    mat.translate(*vloc)
    mat.rotate(vrot[2], 0.0, 0.0, 1.0)
    mat.rotate(vrot[1], 0.0, 1.0, 0.0)
    mat.rotate(vrot[0], 1.0, 0.0, 0.0)
    return mat
