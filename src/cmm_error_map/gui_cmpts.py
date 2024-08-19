import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt as qtc
import pyqtgraph.Qt.QtWidgets as qtw
import pyqtgraph.opengl as gl
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.parametertree import Parameter, ParameterTree

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

# TODO these need to be parameters in gui or config
ballspacing = 133.0

U95 = 1.2

# TODO expand this structure to incldude artefact parameters
default_artefacts = {"MSL Ballplate A": 0}


def single_grid_plot_data(dxy, mag, lines=True, circles=True):
    """
    wrangles the data from the model into the right shape for plotting
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

    if circles:
        # find points outside circles and mark with cross
        ballnumber = np.arange(dxy.shape[0])
        xplaten = (ballnumber) % 5
        yplaten = (ballnumber) // 5
        xcirc = xplaten * ballspacing
        ycirc = yplaten * ballspacing
        rcirc = mag * (U95 + ((xcirc**2 + ycirc**2) ** 0.5) / 400.0) * 1e-3
        err = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2) ** 0.5
        xout = xplot[err > rcirc / mag]
        yout = yplot[err > rcirc / mag]
        data.append((xout, yout))
    return data


def plot_ballplate(params, mag, lines=True, circles=True):
    """
    pyqtgraph 2d pot of ballplate errors
    takes a set of model parameters and produces a 2D magniifed plot of errors in ballplate mmt
    """

    # XZ plane
    RP = np.array(
        [
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 1.0, 50.0],
            [0.0, 1.0, 0.0, 50.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    lineplots = []
    pw = pg.PlotWidget(name="XZ")
    xt, yt, zt = 0.0, 130.0, -243.4852
    dxy = design.modelled_mmts_XYZ(RP, xt, yt, zt, params)
    data = single_grid_plot_data(dxy, mag)
    di = 0
    p1 = pw.plot(
        x=data[di][0],
        y=data[di][1],
        pen=None,
        symbol="o",
    )
    lineplots.append(p1)
    di += 1

    if lines:
        for i in range(0, 5):
            p2 = pw.plot(x=data[di][0], y=data[di][1])
            di += 1
            lineplots.append(p2)
            p3 = pw.plot(x=data[di][0], y=data[di][1])
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

        p4 = pw.plot(
            x=xcirc, y=ycirc, pen=None, symbol="o", symbolSize=rcirc, pxMode=False
        )
        lineplots.append(p4)

        # points outside circles are marked with cross
        p5 = pw.plot(
            x=data[di][0],
            y=data[di][1],
            symbol="x",
            symbolBrush="red",
            symbolSize=10,
            pen=None,
        )
        lineplots.append(p5)
        pw.setAspectLocked()
        grid = pg.GridItem()
        grid.setTickSpacing(x=[ballspacing], y=[ballspacing])
        pw.addItem(grid)

    return pw


class Plot2dDock(Dock):
    """
    a pyqtgraph Dock containing a plot and a side bar with parameter tree controls
    knows how to draw and update itself based on the values in parameter tree
    """

    def __init__(self, name):
        super(Plot2dDock, self).__init__(name)

        self.magnification = 5000
        self.artefacts = default_artefacts

        h_split = qtw.QSplitter(qtc.Horizontal)
        self.plot_params, self.tree = self.make_control_tree()
        self.plot = self.make_plot()
        h_split.addWidget(self.tree)
        h_split.addWidget(self.plot)
        self.addWidget(h_split)

    def make_control_tree(self):
        """
        returns the controls that go in the side bar of each 2d plot
        """
        plot2d_params = Parameter.create(name="params", type="group")
        plot2d_params.addChild(
            dict(
                type="list",
                name="artefact",
                title="artefact type",
                limits=self.artefacts,
            )
        )
        plot2d_params.addChild(grp_position)
        plot2d_params.addChild(grp_plate_dirn)
        plot2d_tree = ParameterTree(showHeader=False)
        plot2d_tree.setParameters(plot2d_params, showTop=False)
        return plot2d_params, plot2d_tree

    def make_plot(self):
        """
        plots the undeformed ballplate ready for update plot
        """
        params0 = np.zeros(21)
        plot2d = plot_ballplate(params0, self.magnification)
        return plot2d

    def update_plot(self, model_params):
        """
        updates the 2d plot with new model parameters from MainWindow model sliders
        """
        pass
