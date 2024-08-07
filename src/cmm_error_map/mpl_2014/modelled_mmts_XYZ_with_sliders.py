# -------------------------------------------------------------------------------
# Name:        modelled_mmts_XYZ_with_sliders
# Purpose:
#
# Author:      e.howick
#
# Created:     12/06/2013
# Copyright:   (c) e.howick 2013
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import numpy as np
import pylab
from matplotlib.widgets import Slider
from design_matrix_linear import modelled_mmts_XYZ, modelparameters
from ballplate_plots import single_grid_plot

ballspacing = 133.0
"""
0	TxxL
1	TxyL
2	TxzL
3	TyxL
4	TyyL
5	TyzL
6	TzxL
7	TzyL
8	TzzL
9	RxxL
10	RxyL
11	RxzL
12	RyxL
13	RyyL
14	RyzL
15	RzxL
16	RzyL
17	RzzL
18	Wxy
19	Wxz
20	Wyz
"""

Tmult = 1e-6
Rmult = 1e-8
Wmult = 1e-8


def main():
    params = np.zeros(21)
    # params[9] = 2e-8 #Rxx
    x0, y0, z0 = 250.0, 50.0, 50.0

    # limits for sliders
    Pmult = np.zeros(21)
    Pmult[:9] = Tmult
    Pmult[9:18] = Rmult
    Pmult[18:] = Wmult

    # XY plane
    x0xy, y0xy, z0xy = x0, y0, z0

    RPXY = np.array(
        [
            [1.0, 0.0, 0.0, x0xy],
            [0.0, 1.0, 0.0, y0xy],
            [0.0, 0.0, 1.0, z0xy],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    xt_xy, yt_xy, zt_xy = 0.0, 0.0, -243.4852

    dxy = modelled_mmts_XYZ(RPXY, xt_xy, yt_xy, zt_xy, params)

    plot_xy = pylab.subplot(223)
    pylab.title("XY")
    pylab.xlabel("X")
    pylab.ylabel("Y")
    linesXY = single_grid_plot(dxy, 10000, lines=True, symbol="ro", plabel="XY")

    # XZ plane
    x0xz, y0xz, z0xz = x0, y0 + 2.0 * ballspacing, z0
    RPXZ = np.array(
        [
            [1.0, 0.0, 0.0, x0xz],
            [0.0, 0.0, 1.0, y0xz],
            [0.0, 1.0, 0.0, z0xz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    xt_xz, yt_xz, zt_xz = 0.0, 130.0, -243.4852

    dxz = modelled_mmts_XYZ(RPXZ, xt_xz, yt_xz, zt_xz, params)
    plot_xz = pylab.subplot(221)
    pylab.title("XZ")
    pylab.xlabel("X")
    pylab.ylabel("Z")
    linesXZ = single_grid_plot(dxz, 10000, lines=True, symbol="ro", plabel="XZ")

    # YZ plane
    x0yz, y0yz, z0yz = x0 + 2.0 * ballspacing, y0, z0
    RPYZ = np.array(
        [
            [0.0, 0.0, 1.0, x0yz],
            [1.0, 0.0, 0.0, y0yz],
            [0.0, 1.0, 0.0, z0yz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    xt_yz, yt_yz, zt_yz = 130.0, 0.0, -243.4852

    dyz = modelled_mmts_XYZ(RPYZ, xt_yz, yt_yz, zt_yz, params)
    plot_yz = pylab.subplot(222)
    pylab.title("YZ")
    pylab.xlabel("Y")
    pylab.ylabel("Z")
    linesYZ = single_grid_plot(dyz, 10000, lines=True, symbol="ro", plabel="YZ")

    lines = linesXY + linesXZ + linesYZ

    # SLIDERS

    # Also need a reset button

    # set up grid
    vs = 0.02  # height slider
    vg = 0.03  # vertical gap between sliders
    vge = 0.04  # extra vertical gap between groups of sliders

    hs = 0.1  # length slider
    hg = 0.05  # horizontal gap between sliders

    sliders = []
    v1 = vg
    for i in range(7):
        for j in range(3):
            h1 = 0.5 + (j + 1) * hg + j * hs
            slider_box = [h1, v1, hs, vs]
            ax_slider = pylab.axes(slider_box)
            paramid = (6 - i) * 3 + j
            sliders.append(
                Slider(
                    ax_slider,
                    modelparameters[paramid][0],
                    -5.0,
                    5.0,
                    valinit=params[paramid] / Pmult[paramid],
                )
            )

        v1 = v1 + vs + vg
        if i in (0, 3):
            v1 = v1 + vge

    # need a list of 21 callback functions
    # function to make functions
    def MakeUpdateFunc(i):
        def update(val):
            change_param(val, i)

        return update

    # create a list of functions
    updates = []
    for i in range(21):
        updates.append(MakeUpdateFunc(i))

    # connect sliders to functions
    for sid, slider in enumerate(sliders):
        paramid = (6 - sid // 3) * 3 + sid % 3
        slider.on_changed(updates[paramid])

    # function called when parameter changed
    def change_param(value, pid):
        print(value, pid)
        params[pid] = value * Pmult[pid]
        ##        #XY
        ##        for line in linesXY:
        ##            dxy = modelled_mmts_XYZ(RPXY,xt_xy,yt_xy,zt_xy,params)
        ##            line[0].set_data(dxy.T)
        ##
        ##        #XZ
        ##        for line in linesXZ:
        ##            dxz = modelled_mmts_XYZ(RPXY,xt_xz,yt_xz,zt_xz,params)
        ##            line[0].set_data(dxz.T)
        ##
        ##        #YZ
        ##        for line in linesYZ:
        ##            dyz = modelled_mmts_XYZ(RPXY,xt_yz,yt_yz,zt_yz,params)
        ##            line[0].set_data(dyz.T)
        dxy = modelled_mmts_XYZ(RPXY, xt_xy, yt_xy, zt_xy, params)
        plot_xy = pylab.subplot(223)
        plot_xy.clear()
        pylab.title("XY")
        pylab.xlabel("X")
        pylab.ylabel("Y")
        linesXY = single_grid_plot(dxy, 10000, lines=True, symbol="ro", plabel="XY")

        dxz = modelled_mmts_XYZ(RPXZ, xt_xz, yt_xz, zt_xz, params)
        plot_xz = pylab.subplot(221)
        plot_xz.clear()
        pylab.title("XZ")
        pylab.xlabel("X")
        pylab.ylabel("Z")
        linesXZ = single_grid_plot(dxz, 10000, lines=True, symbol="ro", plabel="XZ")

        dyz = modelled_mmts_XYZ(RPYZ, xt_yz, yt_yz, zt_yz, params)
        plot_yz = pylab.subplot(222)
        plot_yz.clear()
        pylab.title("YZ")
        pylab.xlabel("Y")
        pylab.ylabel("Z")
        linesYZ = single_grid_plot(dyz, 10000, lines=True, symbol="ro", plabel="YZ")

        pylab.draw()

    pylab.show()


if __name__ == "__main__":
    main()
