import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt as qtc
import pyqtgraph.Qt.QtWidgets as qtw
import pyqtgraph.opengl as gl
from pyqtgraph.dockarea.Dock import Dock

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

# 2D Plots

# parameter structures for 2d plot controls
grp_position = {
    "title": "Position",
    "name": "grp_position",
    "type": "group",
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


class Plot2dDock(Dock):
    """
    a pyqtgraph Dock containing a plot and a side bar with parameter tree controls
    knows how to draw an update itself based on the values in parameter tree
    """
    def __init__(self, title):
        super(Plot2dDock, self).__init__()
        h_split = qtw.QSplitter(qtc.Horizontal)
        self.plot_params, self.tree = self.make_control_tree()
        self.plot = 
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
        plot2d_params.addChild(gc.grp_position)
        plot2d_params.addChild(gc.grp_plate_dirn)
        plot2d_tree = ParameterTree(showHeader=False)
        plot2d_tree.setParameters(plot2d_params, showTop=False)
        return plot2d_params, plot2d_tree

    def make_plot(self) -> (list, gl.GLViewWidget):
        """
        plot the 2d deformation of the artefact
        """

        return plotlines, plot3d