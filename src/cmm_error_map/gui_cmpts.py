from dataclasses import dataclass, field

import mathutils as mu
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt.QtCore import Qt as qtc

import cmm_error_map.design_matrix_linear_fixed as design

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


@dataclass
class PlotData2d:
    """
    plot data for a 2d artefact
    """

    title: str = "Plot 0"
    transform_mat: mu.Matrix = mu.Matrix()
    probe_vec: mu.Vector = mu.Vector()
    plot: pg.PlotWidget = None
    lineplots: list = field(default_factory=list[pg.PlotDataItem])


@dataclass
class PlotData3d:
    """
    plot data for a 3d machine deformation
    """

    probe_title: str = "probe 0"
    probe_vec: mu.Vector = mu.Vector()
    plot: gl.GLViewWidget = None
    lineplots: list = field(default_factory=list[gl.GLLinePlotItem])


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
    "title": "3D Plot",
    "type": "group",
    "children": [
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
    "name": "probe",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "probe_name",
            "title": "Probe Name",
            "type": "str",
            "value": "Probe 0",
        },
        {
            "name": "is_plotted",
            "title": "Plot in 3D",
            "type": "bool",
            "value": True,
        },
        grp_probe_lengths,
    ],
}


class Plot3dDock(Dock):
    """
    a pyqtgraph Dock containing a 3D plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    """

    def __init__(self, name, model_params):
        super(Plot3dDock, self).__init__(name)

        self.magnification = 5000
        self.model_params = model_params

        self.plot_data = {}
        self.plot_widget = gl.GLViewWidget()
        self.add_control_tree()

        # undeformed
        plot_model3d(self.plot_widget, col="green")

        h_split = qtw.QSplitter(qtc.Horizontal)
        h_split.addWidget(self.tree)
        h_split.addWidget(self.plot_widget)
        self.addWidget(h_split)

    def add_control_tree(self):
        """
        adds the controls that go in the side bar of each 3d plot
        """
        self.plot_controls = Parameter.create(**dock3d_control_grp)

        slider_mag = self.plot_controls.child("slider_mag")
        slider_mag.sigValueChanged.connect(self.change_magnification)

        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=False)

    def plot_ball_plate(self, artefact: dict, plot_data_2d: PlotData2d):
        """
        plots an outline of the given artefact at the transform defined by
        plot_data_2d.transform_mat
        """
        print("plotting ball plate in 3d space")

        xballs = artefact["nballs"][0]
        yballs = artefact["nballs"][0]
        ball_count = xballs * yballs
        ballnumber = np.arange(ball_count)
        xp = (ballnumber % xballs) * ballspacing
        yp = (ballnumber // yballs) * ballspacing
        xyz1 = np.stack((xp, yp, np.zeros_like(xp), np.ones_like(xp)))
        pts = np.array(plot_data_2d.transform_mat) @ xyz1
        col = "red"
        plt = gl.GLScatterPlotItem(pos=pts.T, color=pg.mkColor(col))
        self.plot_widget.addItem(plt)

    def change_magnification(self, control):
        """
        event handler for a change in magnification
        """
        self.magnification = control.value()
        self.update_plot(self.model_params)

    def update_plot(self, model_params: dict):
        """
        updates the 3d plot with new model parameters from MainWindow model sliders
        """
        self.plot_data_from_probe_data()
        self.model_params = model_params
        for plot in self.plot_data.values():
            xt, yt, zt = plot.probe_vec
            update_plot_model3d(
                plot.lineplots,
                self.model_params,
                xt,
                yt,
                zt,
                self.magnification,
            )

    def plot_data_from_probe_data(self):
        """
        create the self.probe_data dict of PlotData3d classes from probe name and vector
        stored in self.probe_dict
        """
        for probe_name, probe in self.probe_data.items():
            if probe_name not in self.plot_data:
                self.plot_data[probe_name] = PlotData3d()
                self.plot_data[probe_name].lineplots = plot_model3d(
                    self.plot_widget, col="blue"
                )
            self.plot_data[probe_name].probe_title = probe["probe_title"]
            self.plot_data[probe_name].probe_vec = probe["probe_vec"]

    def update_probes(self, probe_data):
        """
        called in response to any changes to probes in MainWindow
        """
        self.probe_data = probe_data
        self.plot_data_from_probe_data()
        self.update_plot(self.model_params)


# MARK: 2D Plots

# parameter structures for 2d plot controls
grp_position = {
    "title": "Position",
    "name": "grp_position",
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


default_artefacts = {
    "KOBA 0620": {"ball_spacing": 133, "nballs": (5, 5)},
    "KOBA 0420": {"ball_spacing": 83, "nballs": (5, 5)},
    "KOBA 0320": {"ball_spacing": 60, "nballs": (5, 5)},
}

dock2d_control_grp = {
    "name": "dock2d_control_grp",
    "title": "New Dock",
    "type": "group",
    "children": [
        {
            "type": "str",
            "name": "dock_title",
            "title": "Dock Title",
            "value": "New Dock",
        },
        {
            "name": "artefact",
            "title": "artefact type",
            "type": "list",
            "limits": default_artefacts,
        },
        {
            "type": "slider",
            "name": "slider_mag",
            "title": "Magnification",
            "span": magnification_span,
            "value": 5000,
        },
        {
            "name": "plots_grp",
            "title": "Plots",
            "type": "group",
            "addText": "Add Plot",
            "children": [],
        },
    ],
}

plot2d_control_grp = {
    "title": "Plot 0",
    "name": "plot",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "plot_title",
            "title": "Plot Title",
            "type": "str",
            "value": "Plot 0",
        },
        {
            "name": "probe",
            "title": "Probe",
            "type": "list",
            "limits": [],
        },
        grp_position,
        grp_rotation,
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

    def __init__(self, name, model_params, probe_data):
        super(Plot2dDock, self).__init__(name)

        self.magnification = 5000
        self.model_params = model_params
        self.probe_data = probe_data

        self.plot_data = {}
        self.plot_widget = pg.PlotWidget(name=name)
        self.add_control_tree()

        self.artefact = default_artefacts["KOBA 0620"]

        self.update_plot(self.model_params)

        h_split = qtw.QSplitter(qtc.Horizontal)
        h_split.addWidget(self.tree)
        h_split.addWidget(self.plot_widget)
        self.addWidget(h_split)

    def add_control_tree(self):
        """
        adds the controls that go in the side bar of each 2d plot
        """
        self.plot_controls = Parameter.create(**dock2d_control_grp)
        dock_title = self.plot_controls.child("dock_title")
        dock_title.sigValueChanged.connect(self.change_dock_title)

        plots_grp = self.plot_controls.child("plots_grp")
        self.add_new_plot_grp(plots_grp)
        plots_grp.sigAddNew.connect(self.add_new_plot_grp)

        plots_grp.sigTreeStateChanged.connect(
            lambda: self.update_plot(self.model_params)
        )
        slider_mag = self.plot_controls.child("slider_mag")
        slider_mag.sigValueChanged.connect(self.change_magnification)

        artefact = self.plot_controls.child("artefact")
        artefact.sigValueChanged.connect(self.change_artefact)

        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.plot_controls, showTop=True)

    def add_new_plot_grp(self, parent):
        """
        add the controls for a new plot to the side bar
        """
        new_title = f"Plot {len(parent.childs)}"
        grp_params = plot2d_control_grp.copy()
        grp_params["title"] = new_title
        grp_params["children"][0]["value"] = new_title
        # probe selection drop down should show probe_title but return probe_name
        # probe_title can be changed probe_name can't
        limits = {value["probe_title"]: key for key, value in self.probe_data.items()}
        grp_params["children"][1]["limits"] = limits
        new_grp = parent.addChild(grp_params, autoIncrementName=True)
        new_grp.child("plot_title").sigValueChanged.connect(self.change_plot_title)

    def change_artefact(self, param):
        """
        event handler for a change in artefact
        """
        self.artefact = param.value()
        self.update_plot(self.model_params)

    def change_plot_title(self, param):
        """
        event handler for a change in plot title
        """
        param.parent().setOpts(title=param.value())

    def change_dock_title(self, param):
        """
        event handler for a change in dock title
        """
        param.parent().setOpts(title=param.value())
        self.setTitle(param.value())

    def change_magnification(self, control):
        """
        event handler for a change in magnification
        """
        self.magnification = control.value()
        self.update_plot(self.model_params)

    def update_plot(self, model_params: dict):
        """
        updates the 2d plot with new model parameters from MainWindow model sliders
        """
        self.model_params = model_params
        self.plot_data_from_controls()
        pars = list(model_params.values())
        for plot in self.plot_data.values():
            # convert to parameters required by modelled_mmts_XYZ
            # TODO convert design module to mathutils
            RP = np.array(plot.transform_mat)
            xt, yt, zt = plot.probe_vec
            dxy = design.modelled_mmts_XYZ(
                RP,
                xt,
                yt,
                zt,
                pars,
                ballspacing=self.artefact["ball_spacing"],
                nballs=self.artefact["nballs"],
            )
            data = single_grid_plot_data(
                dxy,
                self.magnification,
                ballspacing=self.artefact["ball_spacing"],
                nballs=self.artefact["nballs"],
            )
            for datum, plot in zip(data, plot.lineplots):
                plot.setData(x=datum[0], y=datum[1])

    def plot_data_from_controls(self):
        """
        create the self.plot_data dict of PlotData classes from current gui settings
        """
        for plot_child in self.plot_controls.child("plots_grp").children():
            name = plot_child.name()
            title = plot_child.title()
            # if an entry doesn't exist create a default one
            if name not in self.plot_data:
                self.plot_data[name] = PlotData2d()
                self.plot_data[name].lineplots = plot_ballplate(self.plot_widget)
            self.plot_data[name].title = title

            euler_deg = plot_child.child("grp_rotation")
            origin = plot_child.child("grp_position")
            probe_name = plot_child.child("probe").value()

            eul = mu.Euler(child_to_vector(euler_deg, mfactor=np.deg2rad(1)))
            mat_rot = eul.to_matrix()
            mat_loc = mu.Matrix.Translation(child_to_vector(origin))

            self.plot_data[name].transform_mat = mat_loc @ mat_rot.to_4x4()
            if probe_name:
                probe = self.probe_data[probe_name]
                self.plot_data[name].probe_vec = probe["probe_vec"]
            else:
                self.plot_data[name].probe_vec = mu.Vector()

    def update_probes(self, probe_data):
        """
        called in response to any changes to probes in MainWindow
        """
        self.probe_data = probe_data
        # update the displayed list of probes for each plot control

        # probe selection drop down should show probe_title but return probe_name
        # probe_title can be changed probe_name can't
        limits = {value["probe_title"]: key for key, value in self.probe_data.items()}
        for plot_child in self.plot_controls.child("plots_grp").children():
            probe = plot_child.child("probe")
            probe.setLimits(limits)

        # update the plots in case the probe lengths changed
        self.update_plot(self.model_params)


def child_to_vector(child: Parameter, mfactor=1.0) -> mu.Vector:
    """
    return the values of child as a mathutils vector
    mfactor can be used to convert from degrees to radians
    mfactor = np.deg2rad(1)
    """
    v = [mfactor * grand_kid.value() for grand_kid in child.children()]
    return mu.Vector(v)
