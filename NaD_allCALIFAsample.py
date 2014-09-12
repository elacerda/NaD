#!/usr/bin/python
import os
import fnmatch
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from pycasso import fitsQ3DataCube
from NaD import line, v0RestFrame, distributionPercentiles
from atpy import Table

#===============================================================================
# morphHubbleType = [
#     'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7',
#     'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm',
#     'I',
# ]
#
# NaD_allDict = {
#     'E0'  : [],
#     'E1'  : [],
#     'E2'  : [],
#     'E3'  : [],
#     'E4'  : [],
#     'E5'  : [],
#     'E6'  : [],
#     'E7'  : [],
#     'S0'  : [],
#     'S0a' : [],
#     'Sa'  : [],
#     'Sab' : [],
#     'Sb'  : [],
#     'Sbc' : [],
#     'Sc'  : [],
#     'Scd' : [],
#     'Sd'  : [],
#     'Sdm' : [],
#     'Sm'  : [],
#     'I'   : [],
# }
#
# A_V_allDict = {
#     'E0'  : [],
#     'E1'  : [],
#     'E2'  : [],
#     'E3'  : [],
#     'E4'  : [],
#     'E5'  : [],
#     'E6'  : [],
#     'E7'  : [],
#     'S0'  : [],
#     'S0a' : [],
#     'Sa'  : [],
#     'Sab' : [],
#     'Sb'  : [],
#     'Sbc' : [],
#     'Sc'  : [],
#     'Scd' : [],
#     'Sd'  : [],
#     'Sdm' : [],
#     'Sm'  : [],
#     'I'   : [],
# }
#===============================================================================

morphHubbleType = [
   'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',  # E(1-7) + S0
   'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S',
   'I',
]

NaD_allDict = {
    'E' : [],
    'S' : [],
    'I' : [],
}

A_V_allDict = {
    'E' : [],
    'S' : [],
    'I' : [],
}

color = {
    'E' : 'r',
    'S' : 'b',
    'I' : 'g',
}

leftOffset = 0
rightOffset = 0
leftOffset = -2
rightOffset = 2

lineEdges = [5880 + leftOffset, 5904 + rightOffset]
lineSideBandsEdges = [5800, 5880, 5906, 5986]
NaD = line('NaD', lineEdges, lineSideBandsEdges)

# for dirpath, dirs, files in os.walk(sys.argv[1]):
#    for filename in fnmatch.filter(files, '*.fits'):

t = Table('/Users/lacerda/dev/astro/AllGalaxies/CALIFA_class_num.fits')
# filesuffix = "_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits"
filesuffix = "_synthesis_eBR_v20_q043.d14a512.ps03.k1.mE.CCM.Bgsd6e.fits"
# filesuffix = "_synthesis_eBR_px1_q043.d14a512.ps03.k1.mE.CCM.Bgsd6e.fits"

for i, (califaID, califaName, Name, RA, DE, hubtyp, bar, merg) in enumerate(t.data):
    fitsFile = '%sK%04d%s' % (sys.argv[1], califaID, filesuffix)
    strHubType = morphHubbleType[hubtyp]
    print fitsFile

    if os.path.isfile(fitsFile):
        K = fitsQ3DataCube(fitsFile)

        O_rf__lz, M_rf__lz, rf_norm__z = v0RestFrame(K, np.interp)

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

        A_V_allDict[strHubType].append(K.A_V.copy())
        NaD_allDict[strHubType].append(NaD_deltaW__z)

f = plt.figure()
plt.xlabel(r'$A_V^{\star}$')
plt.ylabel(r'$\Delta W_{%s}$ [$\AA$]' % NaD.name)

for k in A_V_allDict.keys():
    if len(A_V_allDict[k]) > 0:
        A_V_tmp = np.hstack(np.asarray(A_V_allDict[k]))
        NaD_tmp = np.hstack(np.asarray(NaD_allDict[k]))
        A_V_allDict[k] = A_V_tmp
        NaD_allDict[k] = NaD_tmp

        bins = 20
        q = [5, 50, 95]
        H, xedges, prc = distributionPercentiles(A_V_allDict[k], NaD_allDict[k], bins, q)
        # plt.xlim[-0.1, ])
        plt.scatter(A_V_allDict[k], NaD_allDict[k], marker = 'o', s = 0.1)
        step = (xedges[1] - xedges[0]) / 2.
        p5 = prc[:, 0]
        median = prc[:, 1]
        p95 = prc[:, 2]

        rhoSpearman, pvalSpearman = st.spearmanr(A_V_allDict[k], NaD_allDict[k])
        # txt = r'$\Delta W_{%s}\ \to\ %.2f$ as $A_V^{\star} \ \to\ 0$ ($R_s$: $%.3f$)' % (NaD.name, median[0], rhoSpearman)
        # plt.text(0.10, 0.9, txt, fontsize = 12, backgroundcolor = 'w', transform = plt.gca().transAxes, verticalalignment = 'center', horizontalalignment = 'left')

        plt.plot(xedges[:-1] + step, p5, ls = '--', c = color[k])
        plt.plot(xedges[:-1] + step, median, ls = '-', lw = 2, label = '%s %.2f' % (k, rhoSpearman), c = color[k])
        plt.plot(xedges[:-1] + step, p95, ls = '--', c = color[k])
        plt.legend()
        plt.xlim([0.1, 2.5])
        plt.ylim([-5, 5])

f.savefig('AllGalaxies-AV_DeltaEW_%s.png' % NaD.name)
