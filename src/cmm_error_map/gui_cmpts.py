import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt as qtc
import pyqtgraph.Qt.QtWidgets as qtw
import pyqtgraph.opengl as gl
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.parametertree import Parameter, ParameterTree

import cmm_error_map.design_matrix_linear_fixed as design
from dataclasses import dataclass, field

import mathutils as mu
import cmm_error_map.geometry3d as g3d


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


def plot_model3d(w: gl.GLViewWidget, xt, yt, zt, col="white") -> list:
    """
    produces a 3D magniifed plot of the undeformed machine ready for updating with
    deformation via update_plot_model3d
    """
    params = [0.0] * 21
    XYZ, eXYZ = design.machine_deformation(params, xt, yt, zt)
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


def update_plot_model3d(plot_lines: list, params: dict, xt, yt, zt, mag):
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
grp_plate_dirn = {
    "title": "Plate Orientation",
    "name": "grp_plate_dirn",
    "type": "group",
    "expanded": False,
    "children": [
        {
            "name": "x-axis",
            "type": "group",
            "children": [
                {"name": "i", "type": "float", "value": 1.0},
                {"name": "j", "type": "float", "value": 0.0},
                {"name": "k", "type": "float", "value": 0.0},
            ],
        },
        {
            "name": "z-axis",
            "type": "group",
            "children": [
                {"name": "i", "type": "float", "value": 0.0},
                {"name": "j", "type": "float", "value": 0.0},
                {"name": "k", "type": "float", "value": 1.0},
            ],
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

# TODO expand this structure to incldude artefact parameters
default_artefacts = {"MSL Ballplate A": 0}

dock_control_grp = {
    "name": "dock_control_grp",
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
        grp_position,
        grp_plate_dirn,
        grp_probe_lengths,
    ],
}


# TODO these need to be parameters in gui or config
ballspacing = 133.0

U95 = 1.2


@dataclass
class PlotData:
    title: str = "Plot 0"
    transform_mat: mu.Matrix = mu.Matrix()
    probe_vec: mu.Vector = mu.Vector()
    plot: pg.PlotWidget = None
    lineplots: list = field(default_factory=list[pg.PlotDataItem])


def single_grid_plot_data(dxy, mag, lines=True, circles=True):
    """
    drawing red crosses on out of tolerance point commented out
    until I figure how to update them (number of red crosses can change)
    dxy shape(25,2) or shape (20,2) single set of data to plot on current figure
    in order of ballnumber
    """
    ballnumber = np.arange(dxy.shape[0])
    xplaten = (ballnumber) % 5
    yplaten = (ballnumber) // 5

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
    lines=True,
    circles=True,
) -> list[pg.PlotDataItem]:
    """
    pyqtgraph 2d pot of ballplate errors
    takes a set of model parameters and produces a 2D magniifed plot of errors in ballplate mmt
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
    xt, yt, zt = 0.0, 0.0, -100.0
    lineplots = []

    mag = 1
    dxy = design.modelled_mmts_XYZ(RP, xt, yt, zt, params)
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

    def __init__(self, name, model_params):
        super(Plot2dDock, self).__init__(name)

        self.magnification = 5000
        self.model_params = model_params

        h_split = qtw.QSplitter(qtc.Horizontal)
        self.plot_controls, self.tree = self.make_control_tree()
        self.plot_data = {}
        self.plot_data_from_controls()

        self.plot_widget = pg.PlotWidget(name=name)

        for plot in self.plot_data.values():
            plot.lineplots = plot_ballplate(self.plot_widget)

        self.update_plot(self.model_params)
        h_split.addWidget(self.tree)
        h_split.addWidget(self.plot_widget)
        self.addWidget(h_split)

    def make_control_tree(self):
        """
        returns the controls that go in the side bar of each 2d plot
        """
        plot_controls = Parameter.create(**dock_control_grp)
        dock_title = plot_controls.child("dock_title")
        dock_title.sigValueChanged.connect(self.change_dock_title)

        plots_grp = plot_controls.child("plots_grp")
        self.add_new_plot_grp(plots_grp)
        plots_grp.sigAddNew.connect(self.add_new_plot_grp)

        # plot_controls.sigTreeStateChanged.connect(self.update_plot_controls)
        slider_mag = plot_controls.child("slider_mag")
        slider_mag.sigValueChanged.connect(self.change_magnification)

        plot2d_tree = ParameterTree(showHeader=False)
        plot2d_tree.setParameters(plot_controls, showTop=True)
        return plot_controls, plot2d_tree

    def add_new_plot_grp(self, parent):
        """
        add the controls for a new plot to the side bar
        """
        new_title = f"Plot {len(parent.childs)}"
        grp_params = plot2d_control_grp.copy()
        grp_params["title"] = new_title
        grp_params["children"][0]["value"] = new_title
        new_grp = parent.addChild(grp_params, autoIncrementName=True)
        new_grp.child("plot_title").sigValueChanged.connect(self.change_plot_title)

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

    def update_plot_controls(self, group, changes):
        """
        event handler for a change in controls for this dock
        """
        control_name = changes[0][0].name()
        control_value = changes[0][2]
        if control_name == "slider_mag":
            self.magnification = control_value
            self.update_plot(self.model_params)

    def change_magnification(self, control):
        self.magnification = control.value()
        self.update_plot(self.model_params)

    def update_plot(self, model_params: dict):
        """
        updates the 2d plot with new model parameters from MainWindow model sliders

        RP = [[x5,x20,xn,x0],
           [y5,y20,yn,y0],
           [z5,z20,zn,z0],
           [0,0,0, 1]]

        where (x5,y5,z5) is the direction of the vector from ball 1 to ball 5 (plate X axis)
            (x20,y20,z20) is the direction of the vector from ball 1 to ball 20 (plate Y axis)
            (xn,yn,zn) is the direction perpendicular to the plate (plate Z axis)
            (x0,y0,z0) is the machine position of ball 1
        """
        self.model_params = model_params
        self.plot_data_from_controls()
        pars = list(model_params.values())
        for plot in self.plot_data.values():
            # convert to parameters required by modelled_mmts_XYZ
            # TODO convert design module to mathutils
            RP = np.array(plot.transform_mat)
            xt, yt, zt = plot.probe_vec
            dxy = design.modelled_mmts_XYZ(RP, xt, yt, zt, pars)
            data = single_grid_plot_data(dxy, self.magnification)
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
            self.plot_data[name] = self.plot_data.get(name, PlotData)
            self.plot_data[name].title = title

            xaxis = plot_child.child("grp_plate_dirn", "x-axis")
            zaxis = plot_child.child("grp_plate_dirn", "z-axis")
            origin = plot_child.child("grp_position")
            probe = plot_child.child("grp_probe_lengths")

            self.plot_data[name].transform_mat = g3d.matrix_xz_vector(
                child_to_vector(origin),
                child_to_vector(xaxis),
                child_to_vector(zaxis),
            )
            self.plot_data[name].probe_vec = child_to_vector(probe)
            # print(f"{self.plot_data[name].transform_mat =}")


def child_to_vector(child: Parameter) -> mu.Vector:
    """
    return the values of child as a mathutils vector
    """
    v = [grand_kid.value() for grand_kid in child.children()]
    return mu.Vector(v)
