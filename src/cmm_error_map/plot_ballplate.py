"""
standalaone pyqtgraph app to plot the 3d deformation of a CMM volume
for a linear model of the 21 error parameters
"""

import numpy as np

import pyqtgraph as pg


import cmm_error_map.design_matrix_linear_fixed as design
from pyqtgraph.Qt import QtWidgets

ballspacing = 133.0

U95 = 1.2


def single_grid_plot_data(dxy, mag, lines=True, circles=True):
    """
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


app = pg.mkQApp("Ballplate Errors")

params0 = np.zeros(21)
params = np.zeros(21)
params[0] = 1.33e-05
params[3] = -1.12e-05
params[4] = -5.09e-06
params[6] = 2.6e-05
params[7] = 4.6e-06
params[8] = 3.34e-08
params[9] = 7.49e-09
params[10] = 1.54e-08
params[11] = 5e-09
params[12] = -4.58e-09
params[13] = -1.43e-08
params[14] = 2.19e-08
params[15] = 2.49e-09
params[16] = -7.94e-10
params[17] = 4.78e-08

mag = 10000
mw = QtWidgets.QMainWindow()
mw.resize(800, 800)
cw = QtWidgets.QWidget()
mw.setCentralWidget(cw)
layout = QtWidgets.QVBoxLayout()
cw.setLayout(layout)


pw = plot_ballplate(params, mag)
layout.addWidget(pw)
mw.show()


if __name__ == "__main__":
    pg.exec()
