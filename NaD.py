#!/usr/bin/python
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from pycasso import fitsQ3DataCube

def add_subplot_axes(ax, rect, axisbg = 'w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg = axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize = x_labelsize)
    subax.yaxis.set_tick_params(labelsize = y_labelsize)

    return subax

class line:
    def __init__(self, name, lineExtents = [], sideBandExtents = []):
        self.name = name
        self.lineExtents = lineExtents
        self.sideBandExtents = sideBandExtents

    def get_lineEdge_low(self):
        return self.lineExtents[0]

    def get_lineEdge_top(self):
        return self.lineExtents[1]

    def get_sideBandLeftEdge_low(self):
        return self.sideBandExtents[0]

    def get_sideBandLeftEdge_top(self):
        return self.sideBandExtents[1]

    def get_sideBandRightEdge_low(self):
        return self.sideBandExtents[2]

    def get_sideBandRightEdge_top(self):
        return self.sideBandExtents[3]

def v0RestFrame(K, f_interp):
    c = 299792.5  # km/s

    # Rest-frame spectra
    O_rf__lz = np.zeros((K.Nl_obs, K.N_zone))
    M_rf__lz = np.zeros((K.Nl_obs, K.N_zone))

    # Rest-frame wavelength
    l_rf__lz = (K.l_obs * np.ones((K.N_zone, K.Nl_obs))).T / (1 + K.v_0 / c)

    # linear-interpolating new l_obs fluxes
    # from this point afterward in the code one can use O_RF and M_RF with K.l_obs instead K.f_obs and K.f_syn.
    for z in range(K.N_zone):
        O_rf__lz[:, z] = f_interp(K.l_obs, l_rf__lz[:, z], K.f_obs[:, z])
        M_rf__lz[:, z] = f_interp(K.l_obs, l_rf__lz[:, z], K.f_syn[:, z])

    rf_norm__z = np.median(O_rf__lz[~((K.l_obs < 5590) | (K.l_obs > 5680)), :], axis = 0)

    return O_rf__lz, M_rf__lz, rf_norm__z

# distributionPercentiles() divides X in bins with edges XEDGES, using
# numpy.histogram(). After that, uses the intervals in X as a MASK to calculate
# the percentiles of distribution of Y in this X interval.
def distributionPercentiles(x, y, bins, q):
    H, xedges = np.histogram(x, bins = bins)

    prc = np.zeros((bins, len(q)))

    for i, xe in enumerate(xedges[1:]):
        xe_l = xedges[i]
        xe_r = xe

        mask = (x > xe_l) & (x < xe_r)
        ym = y[mask]

        if len(ym) == 0:
            prc[i, :] = np.asarray([np.nan, np.nan, np.nan])
        else:
            prc[i, :] = np.percentile(ym, q = q)

    return H, xedges, prc

if __name__ == '__main__':
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

    # Calculate the distribution percentiles
    bins = 20
    q = [5, 50, 95]
    H, xedges, prc = distributionPercentiles(K.A_V, NaD_deltaW__z, bins, q)

    f = plt.figure()
    plt.title(r'%s - %s' % (K.galaxyName, K.califaID))
    plt.xlabel(r'$A_V^{\star}$')
    plt.ylabel(r'$\Delta W_{%s}$ [$\AA$]' % NaD.name)
    plt.scatter(K.A_V, NaD_deltaW__z, marker = 'o', s = 0.5)

    p5 = prc[:, 0]
    median = prc[:, 1]
    p95 = prc[:, 2]
    rhoSpearman, pvalSpearman = st.spearmanr(K.A_V, NaD_deltaW__z)
    txt = r'$\Delta W_{%s}\ \to\ %.2f$ as $A_V^{\star} \ \to\ 0$ ($R_s$: $%.3f$)' % (NaD.name, median[0], rhoSpearman)
    plt.text(0.10, 0.9, txt, fontsize = 12, backgroundcolor = 'w', transform = plt.gca().transAxes, verticalalignment = 'center', horizontalalignment = 'left')

    step = (xedges[1] - xedges[0]) / 2.
    plt.plot(xedges[:-1] + step, p5, 'k--')
    plt.plot(xedges[:-1] + step, median, 'k-')
    plt.plot(xedges[:-1] + step, p95, 'k--')

    if len(sys.argv) > 2:
        # Subplot with the Galaxy Image
        subpos = [0.7, 0.55, 0.26, 0.45]
        subax = add_subplot_axes(plt.gca(), subpos)
        plt.setp(subax.get_xticklabels(), visible = False)
        plt.setp(subax.get_yticklabels(), visible = False)
        galimg = plt.imread(sys.argv[2])
        subax.imshow(galimg)

    f.savefig('%s-AV_DeltaEW_%s.png' % (K.califaID, NaD.name))
