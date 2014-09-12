#!/usr/bin/python
import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pycasso import fitsQ3DataCube
from NaD import line, v0RestFrame

fitsfile = sys.argv[1]
# Load FITS
K = fitsQ3DataCube(fitsfile)

O_rf__lz, M_rf__lz, rf_norm__z = v0RestFrame(K, np.interp)

leftOffset = 0
rightOffset = 0
# leftOffset = -2
# rightOffset = 2

lineEdges = [5880 + leftOffset, 5904 + rightOffset]
lineSideBandsEdges = [5800, 5880, 5906, 5986]
NaD = line('NaD', lineEdges, lineSideBandsEdges)

# NaD Masks
NaD_mask = ((K.l_obs < NaD.get_lineEdge_low()) | (K.l_obs > NaD.get_lineEdge_top()))
NaDSideBandLow_mask = ((K.l_obs < NaD.get_sideBandLeftEdge_low()) | (K.l_obs > NaD.get_sideBandLeftEdge_top()))
NaDSideBandUp_mask = ((K.l_obs < NaD.get_sideBandRightEdge_low()) | (K.l_obs > NaD.get_sideBandRightEdge_top()))
NaDSideBands_mask = NaDSideBandLow_mask & NaDSideBandUp_mask

NaDSideBandsAndLine_mask = ((K.l_obs < NaD.get_sideBandLeftEdge_low()) | (K.l_obs > NaD.get_sideBandRightEdge_top()))

# Median flux in every zone spectra
NaDSideBands_median__z = np.median(M_rf__lz[~NaDSideBands_mask, :], axis = 0)

# NaD Delta EW
l_step = 2
NaD_deltaW__z = ((O_rf__lz[~NaD_mask] - M_rf__lz[~NaD_mask]) * l_step).sum(axis = 0) / NaDSideBands_median__z

l_obs = K.l_obs[~NaDSideBandsAndLine_mask]
f_obs = O_rf__lz[~NaDSideBandsAndLine_mask] / K.fobs_norm
f_syn = M_rf__lz[~NaDSideBandsAndLine_mask] / K.fobs_norm
f_res = f_obs - f_syn

max_f_obs = f_obs.max()
max_f_syn = f_syn.max()
max_f_res = f_res.max()
min_f_obs = f_obs.min()
min_f_syn = f_syn.min()
min_f_res = f_res.min()

max_f = max_f_obs

if max_f_syn > max_f:
    max_f = max_f_syn

if max_f_res > max_f:
    max_f = max_f_res

min_f = min_f_res

if min_f_obs < min_f:
    min_f = min_f_obs

if min_f_syn < min_f:
    min_f = min_f_syn

max_f = 1.5
min_f = -0.5

for i, z in enumerate(np.argsort(K.v_0)):
    f = plt.figure()
    plt.title(r'%s - %s - zone %4d - $\Delta W_{%s} = %.2f$ - $A_V^\star = %.2f$' % (K.galaxyName, K.califaID, z, NaD.name, NaD_deltaW__z[z], K.A_V[z]))
    plt.xlim([NaD.get_sideBandLeftEdge_low() - 2, NaD.get_sideBandRightEdge_top() + 2])
    plt.ylim([min_f, max_f])
    plt.xlabel(r'$\lambda$ $[\AA]$')
    
    txt = r'$\sigma_\star$: %.2f km/s - $v_\star$: %.2f km/s' % (K.v_d[z], K.v_0[z])
    plt.text(0.5, 0.05, txt,
             fontsize = 12,
             backgroundcolor = 'w',
             transform = plt.gca().transAxes,
             verticalalignment = 'center',
             horizontalalignment = 'left')
    
    plt.plot(l_obs, f_obs[:, z], label = r'$O_\lambda$')
    plt.plot(l_obs, f_syn[:, z], label = r'$M_\lambda$')
    plt.plot(l_obs, f_res[:, z], label = r'$R_\lambda$')
    
    x = K.l_obs[~NaD_mask]
    y1 = O_rf__lz[~NaD_mask, z] / K.fobs_norm[z]
    y2 = M_rf__lz[~NaD_mask, z] / K.fobs_norm[z]
    plt.fill_between(x, y1, y2, where = y1 >= y2, edgecolor = 'gray', facecolor = 'lightgray', interpolate = True)
    plt.fill_between(x, y1, y2, where = y1 <= y2, edgecolor = 'gray', facecolor = 'darkgray', interpolate = True)
    
    plt.axvline(x = NaD.get_lineEdge_low())
    plt.axvline(x = NaD.get_lineEdge_top())
    plt.axhline(y = NaDSideBands_median__z[z] / K.fobs_norm[z])
    plt.axvline(x = 5893, ls = ':')
    plt.legend()
    
    nMinorTicks = 4 + (NaD.get_sideBandRightEdge_top() - NaD.get_sideBandLeftEdge_low())
    plt.gca().xaxis.set_minor_locator(mpl.ticker.MaxNLocator(nbins = nMinorTicks))
    
    plt.grid()
    f.savefig('%s-%s-%04d.png' % (K.califaID, NaD.name, i))
    plt.close(f)
