# -------------------------------------------------------------------------------
# Name:        ballplate_plots.py
# Purpose:      Various routines for plotting ballplate measureements and derived CMM errors
#
# Author:      e.howick
#
# Created:     22/04/2013
# Copyright:   (c) e.howick 2013
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import pickle
import numpy as np
import pylab
from matplotlib.patches import Circle


ballspacing = 133.0

U95 = 1.2


def plotvsballnumber(
    pos, ballplatedataE, ballplatedataM, fittedvalues, axes=(0, 1, 2), lm="-"
):
    """
    for a position given as integer 1 to 6
    plots EX, EY, EZ data given ballplatedataE
    also plots means from ballplatedataM
    and fitted values
    """
    bd = ballplatedataE
    bM = ballplatedataM
    subarray = bd[bd[:, 0] == pos]
    fvpos = []
    fvpos.append(fittedvalues[:130][bM[:, 0] == pos])
    fvpos.append(fittedvalues[130:][bM[:, 0] == pos])
    fvpos.append(np.zeros_like(fvpos[0]))
    axescolors = ("r", "b", "g")
    for axis in axes:
        pylab.plot(subarray[:, 3], subarray[:, 13 + axis], axescolors[axis] + "+")
        pylab.plot(
            bM[bM[:, 0] == pos][:, 1],
            bM[bM[:, 0] == pos][:, 3 + axis],
            axescolors[axis] + "o",
        )
        pylab.plot(bM[bM[:, 0] == pos][:, 1], fvpos[axis], axescolors[axis] + lm)
    pylab.axis([0, 25, -0.01, 0.01])
    return


def all_plotvsballnumber(dE, dM, fv, axes=(0, 1)):
    for i in range(1, 7):
        pylab.figure(i)
        plotvsballnumber(i, dE, dM, fv, axes=(0, 1))


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


def single_grid_plot_only(data, mag, lines=True, symbol="bo", plabel="", circles=True):
    di = 0
    lineplots = []
    L1 = pylab.plot(data[di][0], data[di][1], symbol, label=plabel)
    lineplots.append(L1)
    di = di + 1

    if lines:
        for i in range(0, 5):
            L2 = pylab.plot(data[di][0], data[di][1], symbol[0] + "-")
            di = di + 1
            lineplots.append(L2)
            L3 = pylab.plot(data[di][0], data[di][1], symbol[0] + "-")
            di = di + 1
            lineplots.append(L3)
    if circles:
        # circles don't need updating
        # draw uncertainty circles at nominal positions
        ballnumber = np.arange(data[0][0].shape[0])
        xplaten = (ballnumber) % 5
        yplaten = (ballnumber) // 5
        xcirc = xplaten * ballspacing
        ycirc = yplaten * ballspacing
        rcirc = mag * (U95 + ((xcirc**2 + ycirc**2) ** 0.5) / 400.0) * 1e-3
        circs = [Circle((xcirc[i], ycirc[i]), rcirc[i]) for i in ballnumber]
        for c in circs:
            c.set_alpha(0.1)
            pylab.gca().add_artist(c)
        # points outside circles are marked with cross
        L4 = pylab.plot(
            data[di][0],
            data[di][1],
            symbol[0] + "x",
            label="out of tol.",
            markersize=10,
        )
        lineplots.append((L4))
        pylab.grid(True)
    pylab.xticks(range(0, 5 * 133, 133))
    pylab.yticks(range(0, 5 * 133, 133))
    pylab.axis("equal")
    pylab.axis([-100, 600, -100, 600])

    return lineplots


def single_grid_plot(dxy, mag, lines=True, symbol="bo", plabel="", circles=True):
    data1 = single_grid_plot_data(dxy, mag, lines=lines, circles=circles)
    lineplots = single_grid_plot_only(
        data1, mag, lines=lines, symbol=symbol, plabel=plabel, circles=circles
    )
    return lineplots


##def single_grid_plot(dxy,mag, lines=True, symbol='bo',plabel='', circles=True):
##    """
##    dxy shape(25,2) or shape (20,2) single set of data to plot on current figure
##    in order of ballnumber
##    """
##    ballnumber = np.arange(dxy.shape[0])
##    xplaten = (ballnumber ) % 5
##    yplaten = (ballnumber ) // 5
##
##    xplot = mag * dxy[:,0] + xplaten * ballspacing
##    yplot = mag * dxy[:,1] + yplaten * ballspacing
##
##    lines = []
##    L1 = pylab.plot(xplot,yplot, symbol,label=plabel)
##    lines.append(L1)
##
##    if lines:
##        for i in range(0,5):
##            L2 = pylab.plot(xplot[xplaten[:]==i], yplot[xplaten[:]==i], symbol[0]+'-')
##            L3 = pylab.plot(xplot[yplaten[:]==i], yplot[yplaten[:]==i], symbol[0]+'-')
##            lines.extend((L2, L3))
##
##    if circles:
##        #draw uncertainty circles at nominal positions
##        xcirc = xplaten * ballspacing
##        ycirc = yplaten * ballspacing
##        rcirc = mag*(U95 + ((xcirc**2 + ycirc**2)**0.5)/400.0)*1e-3
##        circs = [Circle((xcirc[i], ycirc[i]), rcirc[i]) for i in ballnumber]
##        for c in circs:
##            c.set_alpha(0.1)
##            pylab.gca().add_artist(c)
##        #find points outside circles and mark with cross
##        err = (dxy[:,0]**2 + dxy[:,1]**2)**0.5
##        xout = xplot[err > rcirc/mag]
##        yout = yplot[err > rcirc/mag]
##        pylab.plot(xout,yout, symbol[0]+'x',label='out of tol.',markersize=10)
##
##
##    pylab.grid(True)
##    pylab.xticks(range(0,5*133,133))
##    pylab.yticks(range(0,5*133,133))
##    pylab.axis('equal')
##    pylab.axis([-100,600,-100,600])
##
##    return lines


def gridplot(pos, ballplatedataE, fittedvalues, mag, mmtinfo):
    """
    for a position given as integer 1 to 6
    plots measured and fitted values as a magnified grid plot on plate axes
    """
    ballnumber = mmtinfo[(mmtinfo[:, 0] == pos) & (mmtinfo[:, 2] == 0), 1]

    # fitted values
    xfv = fittedvalues[(mmtinfo[:, 0] == pos) & (mmtinfo[:, 2] == 0)]
    yfv = fittedvalues[(mmtinfo[:, 0] == pos) & (mmtinfo[:, 2] == 1)]
    dxy = np.vstack((xfv, yfv)).T

    single_grid_plot(dxy, mag, lines=True, symbol="bo")

    # measured errors
    mpoints = ballplatedataE[ballplatedataE[:, 0] == pos, :]

    sets = set(mpoints[:, 1])
    for s in sets:
        dxy1 = mpoints[mpoints[:, 1] == s, :]
        dxy1 = dxy1[:, (3, 13, 14)]
        # sort on ballnumber
        dxy2 = dxy1[dxy1[:, 0].argsort()]
        dxy3 = dxy2[:, 1:]
        single_grid_plot(dxy3, mag, lines=False, symbol="g+")
    return


def all_gridplot(dE, dM, fv, mag, mmtinfo):
    for i in range(1, 7):
        pylab.figure(i + 6)
        gridplot(i, dE, fv, mag, mmtinfo)


def main():
    pfname = (
        r"L:\CMM Leitz\Ballplate_MSL\Python_analysis\ballplate_linear_solutions.pickle"
    )
    fin = open(pfname, "rb")
    d, dtstime, info, dT, c, dE, dM, mmtinfo, dmatrix, y, all_results = pickle.load(fin)
    fin.close()

    # plot everything!!!

    # fitted values for each results solution are equivalent so only plot fitted values from 21 parameter result
    ##    all_plotvsballnumber(dE, dM, all_results[0][0].fittedvalues, axes=(0,1))
    ##    all_gridplot(dE,dM, all_results[0][0].fittedvalues,10000,mmtinfo)

    dxy = dM[0:25, 3:5]
    ##    data1 = single_grid_plot_data(dxy,5000, lines=True, circles=True)
    ##    single_grid_plot_only(data1, 5000, lines=True, symbol='bo',plabel='', circles=True)

    ##    pylab.figure()
    ##    gridplot(1,dE,all_results[0][0].fittedvalues,10000,mmtinfo)

    lineplots = single_grid_plot(
        dxy, 10000, lines=True, symbol="bo", plabel="test", circles=True
    )

    pylab.show()


if __name__ == "__main__":
    main()
