#-------------------------------------------------------------------------------
# Name:        test_modelled_mmts_XYZ
# Purpose:
#
# Author:      e.howick
#
# Created:     12/06/2013
# Copyright:   (c) e.howick 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pylab
from design_matrix_linear import modelled_mmts_XYZ
from ballplate_plots import single_grid_plot


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




def main():
    #XZ plane
    RP = np.array([[1.0,0.0,0.0,100.0],
                   [0.0,0.0,1.0,50.0],
                   [0.0,1.0,0.0,50.0],
                   [0.0,0.0,0.0,1.0]])

    params = np.zeros(21)
    params[19]  = 5.0e-6  #Txx

    xt,yt,zt = 0.0, 130.0, -243.4852

    dxy = modelled_mmts_XYZ(RP,xt,yt,zt,params)
    single_grid_plot(dxy,10000,lines=True,symbol='ro',plabel='test')
    pylab.show()

if __name__ == '__main__':
    main()
