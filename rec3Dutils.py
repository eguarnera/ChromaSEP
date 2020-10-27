# Collection of functions for reconstruction of 3D embedding of structures
#  from a given distance matrix

# Import
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import hicutils as hcu
#import plotutils as plu
#import networkx as nx
#import seaborn as sns
import pandas as pd
#from py2cytoscape.data.cynetwork import CyNetwork
#from py2cytoscape.data.cyrest_client import CyRestClient
#from py2cytoscape.data.style import StyleUtil, Style
#import py2cytoscape.util.cytoscapejs as cyjs
#import requests
from scipy.linalg import eig, eigvals, eigh, eigvalsh
#import pymol
#from pymol import cmd, util
from scipy.spatial.distance import pdist
import rmsd
from time import sleep
from subprocess import call


def _enforce_triangleInequality(distmat):
    """
    Enforce triangle inequality on distance matrix.
    NaN distance values will be replaced with infinity.
    """
    nbins = len(distmat)
    d = distmat.copy()
    d[np.isnan(d)] = np.inf
    for i in range(nbins):
        b = np.tile(distmat[i], (nbins, 1))
        b = b + b.T
        dp = np.minimum(d, b)
        if not np.allclose(d, dp):
            #print 'Merge at node', i
            pass
        d = dp
    return d


def _enforce_triangleInequalityV2(distmat):
    """
    Enforce triangle inequality on distance matrix.
    """
    nbins = len(distmat)
    d = distmat.copy()
    for i in range(nbins):
        b = np.tile(d[i], (nbins, 1))
        b = b + b.T
        d = np.minimum(d, d[i][np.newaxis, :] + d[i][:, np.newaxis])
    return d


def _enforce_triangleInequalityV2_old(distmat):
    """
    Enforce triangle inequality on distance matrix.
    """
    nbins = len(distmat)
    d = distmat.copy()
    for i in range(nbins):
        b = np.tile(d[i], (nbins, 1))
        b = b + b.T
        dp = np.minimum(d, b)
        if not np.allclose(d, dp):
            #print 'Merge at node', i
            pass
        d = dp
    return d


def _calc_CGCoordinates2(distmat):
    """
    Given distance matrix, compute squared-distances of nodes from CG.
    """
    nbins = len(distmat)
    sumd2 = np.sum([np.sum(np.diag(distmat ** 2, i))
                    for i in range(1, nbins)]) / nbins ** 2
    d02 = np.average(distmat ** 2, axis=1) - sumd2
    return d02


def _distmat2MetricEigs_nD_v2(distmat, n, retAll=False, checkD02=True,
                ignoreNegD02=True, ignoreNegMmat=False, assumeSymm=True,
                takePosEvals=False):
    """
    From distance matrix, compute n-dimensional embedding eigensystem.
    Returns the first n eigenvalues and coordinates when retAll = False.
    If retAll = True, return all eigenvalues and coordinates.
    If checkD02 = True, return error if squared-distances from CG is negative
    for any node. Otherwise, set negative distances to 0.0
    If ignoreNegD02 = True and checkD02 = False, proceed with computation
    without setting begative values of d02 to 0.0.
    """
    nbins = len(distmat)
    d02 = _calc_CGCoordinates2(distmat)
    if (checkD02 and np.sum(d02 < 0.0) > 0):
        print 'Error: Invalid distances from CG d02!'
        return
    else:
        if not ignoreNegD02:
            d02[d02 < 0.0] = 0.0
    metricmat = (d02[:, np.newaxis] + d02[np.newaxis, :] - distmat ** 2) / 2.0
    if assumeSymm and not np.allclose(metricmat, metricmat.T):
        #print 'Error: metricmat is not symmetric!'
        return
    if not ignoreNegMmat:
        metricmat[metricmat < 0.0] = 0.0
    evals, evecs = eigh(metricmat,
                    eigvals=([nbins-3, nbins-1] if takePosEvals else None)) if assumeSymm \
                    else eig(metricmat)
    evalorder = (np.argsort(evals)[::-1] if takePosEvals else
                  np.argsort(np.abs(evals))[::-1])
    inds = evalorder if retAll else evalorder[:n]
    evals2 = evals[inds]
    evecs2 = evecs[:, inds]
    #for i, v in enumerate(evecs2.T):
        #evecs2[:, i] /= np.sum(np.abs(v) ** 2)
    return (evals2, evecs2)


def _distmat2MetricEigs_nD(distmat, n, retAll=False, checkD02=True,
                ignoreNegD02=True, ignoreNegMmat=False, assumeSymm=True,
                takePosEvals=False):
    """
    From distance matrix, compute n-dimensional embedding eigensystem.
    Returns the first n eigenvalues and coordinates when retAll = False.
    If retAll = True, return all eigenvalues and coordinates.
    If checkD02 = True, return error if squared-distances from CG is negative
    for any node. Otherwise, set negative distances to 0.0
    If ignoreNegD02 = True and checkD02 = False, proceed with computation
    without setting begative values of d02 to 0.0.
    """
    nbins = len(distmat)
    d02 = _calc_CGCoordinates2(distmat)
    if (checkD02 and np.sum(d02 < 0.0) > 0):
        print 'Error: Invalid distances from CG d02!'
        return
    else:
        if not ignoreNegD02:
            d02[d02 < 0.0] = 0.0
    d02i = np.tile(d02, (nbins, 1))
    metricmat = (d02i + d02i.T - distmat ** 2) / 2.0
    if assumeSymm and not np.allclose(metricmat, metricmat.T):
        #print 'Error: metricmat is not symmetric!'
        return
    if not ignoreNegMmat:
        metricmat[metricmat < 0.0] = 0.0
    evals, evecs = eigh(metricmat) if assumeSymm else eig(metricmat)
    evalorder = (np.argsort(evals)[::-1] if takePosEvals else
                  np.argsort(np.abs(evals))[::-1])
    inds = evalorder if retAll else evalorder[:n]
    evals2 = evals[inds]
    evecs2 = evecs[:, inds]
    #for i, v in enumerate(evecs2.T):
        #evecs2[:, i] /= np.sum(np.abs(v) ** 2)
    return (evals2, evecs2)


## Matrix norm modes

def _matrixNormalization(mat, mode=0, sigmaratio=1.0e-4, nitermax=10000):
    """
    Attempts to perform row- and column-normalization on matrix mat
    Input mode:
        0: Don't normalize
        1: Perform 'sqrt-vanilla' normalization once
        2: Iterate 'sqrt-vanilla' normalization until sigma/mean of rowsums
           is less than sigmaratio.
           If number of iterations exceeds nitermax, return error.
        3: Perform 'vanilla' normalization once to obtain affinity matrix
    """
    if mode == 0:
        return mat
    elif mode == 1:
        rowsum = np.sqrt(np.sum(mat, axis=0))
        return mat / np.outer(rowsum, rowsum)
    elif mode == 2:
        mat2 = mat.copy()
        niter = 0
        while niter < nitermax:
            rowsum = np.sqrt(np.sum(mat2, axis=0))
            mat2 /= np.outer(rowsum, rowsum)
            rowsum2 = np.sum(mat2, axis=0)
            if np.std(rowsum2) < np.average(rowsum2) * 1.0e-4:
                return mat2
            niter += 1
        print 'Matrix normalization not converging: niters exceeded %i!' % \
                        nitermax
        return
    elif mode == 3:
        rowsum = (np.sum(mat, axis=0))
        return mat / np.outer(rowsum, rowsum)


# Computing / visualizing embedding


def _calc_psize_blobrad(nodedata, edgedata, radius_exp,
            alpha=sys.float_info.epsilon, fab=None):
    """
    Calculate blob radii as a function of partition size.
    """
    nbins = len(nodedata)
    nlbls = list(nodedata.index)
    if fab is None:
        fab = np.ones((nbins, nbins))
        for i1, n1 in enumerate(nlbls):
            for i2, n2 in enumerate(nlbls):
                if i1 >= i2:
                    continue
                fab[i1, i2] = fab[i2, i1] = edgedata[(edgedata['Lbl1'] == n1) &
                                (edgedata['Lbl2'] == n2)]['Fab']
    psizes = list(nodedata['Partition size'])
    dab2norm2 = 1.0 / (_matrixNormalization(fab, mode=2) + alpha)
    dab2norm2 -= np.diag(np.diag(dab2norm2))
    distmat = dab2norm2.copy()
    dsc = np.average(distmat[distmat <= np.percentile(distmat.flatten(), 95)])
    psize_exp = (np.array(psizes) / np.percentile(psizes, 95)) ** \
                    radius_exp
    psc = np.average((psize_exp[psize_exp <=
                    np.percentile(psize_exp, 95)]))
    radius_scale = dsc / psc
    blobrad = psize_exp * radius_scale
    return psizes, blobrad


def _getEmbedding_3d(evals, evecs):
    """
    Takes 3 largest eigenvalues/vectors to construct embedding.
    Assumes there are at least 3 positive eigenvalues.
    """
    inds = np.argsort(np.real(evals))[::-1][:3]
    lvals = np.real(evals[inds])
    vecs = evecs[:, inds]
    if np.sum(lvals < 0.0) > 0:
        print 'Warning: Top 3 eigenvalues are', lvals
        lvals = np.abs(lvals)
    #print 'eigenvalues:', lvals
    for i, lv in enumerate(lvals):
        vecs[:, i] *= np.sqrt(lv) / np.sqrt(np.sum(np.abs(vecs[:, i]) ** 2))
    return vecs


def _vizEmbedding_3D(evals, evecs):
    vecs = _getEmbedding_3d(evals, evecs) / np.sqrt(np.max(evals))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*(vecs.T), marker='o')


def mapVecToColors(vec, cmap, valrange=[-1, 1]):
    """
    Map vector of values onto a colormap over range valrange.
    """
    v = np.array(vec)
    v = (v - valrange[0]) / float(valrange[1] - valrange[0]) * 255.0
    return np.array(map(cmap, map(int, v)))


def _scatterEmbedding_3D(evals, evecs, sizes, maxsize=40, clbls=None):
    vecs = np.real(_getEmbedding_3d(evals, evecs) / np.sqrt(np.max(evals)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sizes_r = (np.array(sizes) / np.mean(sizes)) ** 2.0 * float(maxsize)
    if clbls is None:
        ax.plot(*(vecs.T))
        ax.scatter(*(vecs.T), s=sizes_r)
    else:
        # Get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        nclrs = len(colors)
        # Extract chromosomes
        chrlist, chrdata, sizedata = _extractChrs(vecs, clbls, sizes)
        for i, (cname, vs, ss) in enumerate(zip(chrlist, chrdata, sizedata)):
            ss2 = (np.array(ss) / np.mean(sizes)) ** 2.0 * float(maxsize)
            ax.plot(*(np.array(vs).T), c=colors[i % nclrs])
            ax.scatter(*(np.array(vs).T), s=ss2,
                    c=colors[i % nclrs], label=cname)
        ax.legend()
    ax.set_aspect('equal')


def _extractChrs(vecs, clbls, sizes):
    seen = set()
    seen_add = seen.add
    clist = [x for x in clbls if not (x in seen or seen_add(x))]
    return clist, [vecs[np.array(clbls) == c] for c in clist], \
            [np.array(sizes)[np.array(clbls) == c] for c in clist]


def Embed3D(data, tri_ineq='V2'):
    distmat = data.copy()
    if tri_ineq == 'V2':
        distmat = _enforce_triangleInequalityV2(distmat)
    else:
        distmat = _enforce_triangleInequality(distmat)
    evals, evecs = _distmat2MetricEigs_nD(distmat, 3,
            retAll=True, checkD02=False, ignoreNegD02=False,
            ignoreNegMmat=True, assumeSymm=False)
    evals = np.real(evals)
    _vizEmbedding_3D(evals, evecs)


def Embed3D_scatter(data, sizes=None, tri_ineq='V2'):
    distmat = data.copy()
    if tri_ineq == 'V2':
        distmat = _enforce_triangleInequalityV2(distmat)
    else:
        distmat = _enforce_triangleInequality(distmat)
    evals, evecs = _distmat2MetricEigs_nD(distmat, 3,
            retAll=True, checkD02=False, ignoreNegD02=False,
            ignoreNegMmat=True, assumeSymm=False)
    evals = np.real(evals)
    if sizes is None:
        sizes = 1.0
    _scatterEmbedding_3D(evals, evecs, sizes)


def Embed3D_blobs(data, sizes=None, tri_ineq='V2',
        radius_mode=0, radius_scale=1.0, clbls=None):
    """
    Input:
        - data: Distance matrix
        - sizes: Partition sizes
        - tri_ineq: Type of triangle-inequality smoothing
        - radius_mode: How to define partition blob radius from sizes
                       and radius_scale.
            0: radius = size / maxsize * rscale
            1: radius = (size / maxsize) ** (1/3) * rscale
            2: radius = (size / maxsize) ** (0.6) * rscale
    """
    distmat = data.copy()
    nbins = len(distmat)
    radius_exp = 1.0 if (radius_mode == 0) else \
            (1.0 / 3.0 if (radius_mode == 1) else 0.6)
    if sizes is None:
        sizes = np.ones(nbins)
    blobrad = (sizes / np.percentile(sizes, 95)) ** radius_exp * radius_scale
    d1 = np.tile(blobrad, (nbins, 1))
    d1 = d1 + d1.T
    d1 -= np.diag(np.diag(d1))
    distmat += d1
    if tri_ineq == 'V2':
        distmat = _enforce_triangleInequalityV2(distmat)
    else:
        distmat = _enforce_triangleInequality(distmat)
    evals, evecs = _distmat2MetricEigs_nD(distmat, 3,
            retAll=True, checkD02=False, ignoreNegD02=False,
            ignoreNegMmat=True, assumeSymm=False)
    evals = np.real(evals)
    _scatterEmbedding_3D(evals, evecs, sizes, clbls=clbls)


#################################################

# Coordinate transforms


def _calc_bodyCenterCoordinates(coords, masses=None):
    """
    Calculate body-centered of a distribution of point masses, i.e.,
     with centroid at origin.
    If masses is None, treat points as equal mass.
    Inputs:
    - coords: (N, 3) array of coordinates
    - masses: (N,) array of masses, or None
    """
    if masses is None:
        masses = np.ones(len(coords))
    shift = -np.sum(np.dot(np.diag(masses), coords), axis=0) / np.sum(masses)
    return coords + np.tile(shift, [len(coords), 1])


def _calc_rescaleGyradius(coords, masses=None, retFactor=False):
    """
    Center coords to place centroid at origin, then rescale coords to set
     radius of gyration to 1.0.
    If masses is None, treat points as equal mass.
    Inputs:
    - coords: (N, 3) array of coordinates
    - masses: (N,) array of masses, or None
    """
    coords_ctr = _calc_bodyCenterCoordinates(coords, masses=masses)
    if masses is None:
        gyradius = np.sqrt(np.sum(np.sum(coords_ctr ** 2, axis=1)) /
                        len(coords))
    else:
        gyradius = np.sqrt(np.sum(np.sum(coords_ctr ** 2, axis=1) * masses) /
                        np.sum(masses))
    if retFactor:
        return coords_ctr / gyradius, 1.0 / gyradius
    else:
        return coords_ctr / gyradius


def _calc_principalAxes(coords, masses=None):
    """
    Calculate principal axes of a distribution of point masses.
    If masses is None, treat points as equal mass.
    Inputs:
    - coords: (N, 3) array of coordinates
    - masses: (N,) array of masses, or None
    """
    coords_ctr =_calc_bodyCenterCoordinates(coords, masses=masses)
    # Calculate inertia tensor
    if masses is None:
        imat = np.dot(coords_ctr.T, coords_ctr)
    else:
        imat = np.dot(coords_ctr.T, np.dot(np.diag(masses), coords_ctr))
    # Calculate eigenvalues / vectors
    evals, evecs = eig(imat)
    order = np.argsort(evals)
    eval3, eval2, eval1 = evals[order]
    #print 'evals:', eval1, eval2, eval3
    evec3, evec2, evec1 = evecs[:, order].T
    return evec1, evec2, evec3


def orientCoordsToPrincipalAxes(coords, masses=None):
    """
    Orient coordinates of distribution of point masses such that the
    principal axes line up with x, y, z-axes.
    Inputs:
    - coords: (N, 3) array of coordinates
    - masses: (N,) array of masses, or None
    """
    coords_ctr =_calc_bodyCenterCoordinates(coords, masses=masses)
    evecs = np.array(_calc_principalAxes(coords_ctr, masses=None)).T
    return np.dot(coords_ctr, evecs)


##############################################

# Reading saved eigenspace data


def _get_eigendata(key, optionsDf):
    """
    Read eigenspace data from file
    """
    trialDf = pd.DataFrame(index=[0], columns=['InteractionCutoff',
        'DistanceExponent', 'DistanceConstraint', 'RadiusFactor',
        'RadiusExponent', 'TemperatureBeta'])
    trialDf.loc[0] = key
    matchindex = None
    for ind, row in optionsDf.iterrows():
        if np.allclose(key, row):
            matchindex = ind
            break
    #match = optionsDf[((optionsDf == trialDf.loc[0]) |
            #(optionsDf.isnull() & trialDf.loc[0].isnull())).all(1)]
    #if len(match) == 0:
        ## Combination not recorded in optionsDf
        #print 'Eigenspace data not found!'
        #return
    if matchindex is None:
        print 'Eigenspace data not found!'
        return
    #else:
        #matchindex = match.index[0]
    fname = os.path.join('EigenspaceData', '%i.dat' % matchindex)
    return hcu._pickle_secureread(fname, free=True)


####################################################

# File I/O


def _read_xyz(fname):
    """
    Read XYZ coordinates file.
    """
    with open(fname, 'r') as f:
        npts = int(f.readline().split()[0])
        f.readline()
        lbls = []
        coords = []
        for i in range(npts):
            n, x, y, z = f.readline().split()
            x = float(x)
            y = float(y)
            z = float(z)
            lbls.append(n)
            coords.append([x, y, z])
    return lbls, np.array(coords)

####################################################

# Reconstructing perturbed systems


def _test_subsetRecon(fab, coordsr, psizes, beta, rf, radius_exp, sample_frac,
                normd2=True, outputcoords=False):
    """
    Try reconstructing a subset of the genome, get RMSD deviation from
    full-recon structure and selected indices.
    If normd2 is True, normalize coordinates such that rms distance
        from origin is 1.0.
    """
    # Select random slice (fixed fraction)
    nbins = len(fab)
    ssize = int(sample_frac * nbins)
    selinds = np.sort(np.random.choice(nbins, size=ssize, replace=False))
    fmat = fab.copy()[selinds][:, selinds]
    psizesel = np.array(psizes)[selinds]
    # Compute coords, blobrad, dump to file
    coords, blobrads = ShRec3D_blobs(fmat, psizesel, radius_exp, rf, normmode=2,
                tri_ineq='V2', alpha=sys.float_info.epsilon, beta=beta, gamma=1.0)
    coords /= np.sqrt(np.average(coords ** 2))
    # Compute rmsd between full-recon and slice-recon
    ## (only at intersecting parts)
    ## TO TEST
    data = [0 for i in range(8)]
    for i in range(2):
        x = coords[:, 0] * (-1) ** i
        for j in range(2):
            y = coords[:, 1] * (-1) ** j
            for k in range(2):
                z = coords[:, 2] * (-1) ** k
                ind = ((i * 2) + j) * 2 + k
                data[ind] = np.array([x, y, z]).T
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr[selinds]) for c in data]
    igood = np.argmin(rmsdcheck)
    coords = rmsd.kabsch_rotate(data[igood], coordsr[selinds])
    dist = rmsdcheck[igood]
    if outputcoords:
        return dist, selinds, coords
    else:
        return dist, selinds


def _test_MultGaussResampleRecon(fab, coordsr, psizes, beta, rf, radius_exp,
            noise_fracs, normd2=True, outputcoords=False):
    """
    Try reconstructing Poisson-resampled genome-wide interaction matrix,
            get RMSD deviation from base recon structure.
    If normd2 is True, normalize coordinates such that rms distance
        from origin is 1.0.
    """
    noise_size, noise_fab, noise_brad, noise_dist = noise_fracs
    # Resample interaction matrix
    fmat = np.random.normal(fab, fab * noise_fab)
    fmat = (fmat + fmat.T) / 2.0
    #psizes2 = np.random.normal(psizes, np.array(psizes) * noise_size)
    psizes2 = psizes
    # Compute coords, blobrad, dump to file
    if noise_brad > 0.0 or noise_dist > 0.0:
        coords, blobrads = ShRec3D_blobs_noise(fmat, psizes2, radius_exp, rf,
                (noise_brad, noise_dist), normmode=2, tri_ineq='V2',
                alpha=sys.float_info.epsilon, beta=beta, gamma=1.0)
    else:
        coords, blobrads = ShRec3D_blobs(fmat, psizes2, radius_exp, rf,
                    normmode=2, tri_ineq='V2', alpha=sys.float_info.epsilon, beta=beta,
                    gamma=1.0)
    coords /= np.sqrt(np.average(coords ** 2))
    # Compute rmsd between full-recon and slice-recon
    ## (only at intersecting parts)
    ## TO TEST
    data = [0 for i in range(8)]
    for i in range(2):
        x = coords[:, 0] * (-1) ** i
        for j in range(2):
            y = coords[:, 1] * (-1) ** j
            for k in range(2):
                z = coords[:, 2] * (-1) ** k
                ind = ((i * 2) + j) * 2 + k
                data[ind] = np.array([x, y, z]).T
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr) for c in data]
    igood = np.argmin(rmsdcheck)
    coords = rmsd.kabsch_rotate(data[igood], coordsr)
    dist = rmsdcheck[igood]
    if outputcoords:
        return dist, coords
    else:
        return dist


####################################################

# Full computation


def ShRec3D_blobs(fmat, psizes, radius_exp, radius_factor, normmode=2,
        tri_ineq='V2', alpha=sys.float_info.epsilon, beta=1.0, gamma=1.0,
        limitDist='None'):
    """
    Computing 3D structure from interaction matrix, considering finite
    partition radii.

    d_ij = (norm{f_ij^beta} + alpha)^{-gamma}
    r_i ~ rfactor * s_i^rexp
    D_ij = d_ij + r_i + r_j

    limitDist: Setting a cutoff for distance matrix... Possible values include:
        - 'None'  : No cutoff
        - 'logMax': maximum at distribution of log(Dij)
        - 'logAvg': average at distribution of log(Dij)
    """
    distmat = 1.0 / (_matrixNormalization(fmat ** beta, mode=normmode) +
                    alpha) ** gamma
    distmat -= np.diag(np.diag(distmat))
    nbins = len(distmat)
    dsc = np.average(distmat[distmat <= np.percentile(distmat.flatten(), 95)])
    psize_exp = (np.array(psizes) / np.percentile(psizes, 95)) ** \
                    radius_exp
    psc = np.average((psize_exp[psize_exp <=
                    np.percentile(psize_exp, 95)]))
    radius_scale = dsc / psc
    blobrad = psize_exp * radius_scale * radius_factor
    d1 = np.tile(blobrad, (nbins, 1))
    d1 = d1 + d1.T
    d1 -= np.diag(np.diag(d1))
    distmat += d1
    # Limit D_ij?
    if limitDist == 'None':
        pass
    elif limitDist == 'logMax':
        hist, edges = np.histogram(np.log(distmat[distmat > 0.0]), bins=100)
        bestind = np.argmax(hist)
        cutoffval = np.exp(np.average(edges[bestind:bestind + 2]))
        distmat[distmat > cutoffval] = cutoffval
    elif limitDist == 'logAvg':
        cutoffval = np.exp(np.average(np.log(distmat[distmat > 0.0])))
        distmat[distmat > cutoffval] = cutoffval
    else:
        print 'ShRec3D_blobs error: Unknown limitDist, ignoring!'
    # Triangle inequality
    if tri_ineq == 'V2':
        distmat = _enforce_triangleInequalityV2(distmat)
    else:
        distmat = _enforce_triangleInequality(distmat)
    evals, evecs = _distmat2MetricEigs_nD(distmat, 3,
            retAll=False, checkD02=False, ignoreNegD02=False,
            ignoreNegMmat=True, assumeSymm=False)
    evals = np.real(evals)
    coords = np.real(_getEmbedding_3d(evals, evecs))
    return coords, blobrad


def ShRec3D_blobs_noise(fmat, psizes, radius_exp, radius_factor, noise_fracs,
        normmode=2, tri_ineq='V2', alpha=sys.float_info.epsilon, beta=1.0,
        gamma=1.0, limitDist='None'):
    """
    Computing 3D structure from interaction matrix, considering finite
    partition radii.

    d_ij = (norm{f_ij^beta} + alpha)^{-gamma}
    r_i ~ rfactor * s_i^rexp
    D_ij = d_ij + r_i + r_j
    """
    noise_brad, noise_dist = noise_fracs
    nbins = len(fmat)
    distmat = 1.0 / (_matrixNormalization(fmat ** beta, mode=normmode) +
                    alpha) ** gamma
    ### Ignore noise in distance matrix: normalization must be kept!
    ### Implement noise in F-matrix instead
    #if noise_dist > 0.0:
        #distmat = np.random.uniform(distmat, distmat * noise_dist)
        #distmat = (distmat + distmat.T) / 2.0
    distmat -= np.diag(np.diag(distmat))
    psize_exp = (np.array(psizes) / np.percentile(psizes, 95)) ** \
                    radius_exp
    psc = np.average((psize_exp[psize_exp <=
                    np.percentile(psize_exp, 95)]))
    dsc = np.average(distmat[distmat <= np.percentile(distmat.flatten(), 95)])
    radius_scale = dsc / psc
    blobrad = psize_exp * radius_scale * radius_factor
    if noise_brad > 0.0:
        rvec = np.random.normal(1.0, scale=noise_brad, size=len(blobrad))
        blobrad *= rvec
    d1 = np.tile(blobrad, (nbins, 1))
    d1 = d1 + d1.T
    d1 -= np.diag(np.diag(d1))
    distmat += d1
    distmat -= np.diag(np.diag(distmat))
    # Limit D_ij?
    if limitDist == 'None':
        pass
    elif limitDist == 'logMax':
        hist, edges = np.histogram(np.log(distmat[distmat > 0.0]), bins=100)
        bestind = np.argmax(hist)
        cutoffval = np.exp(np.average(edges[bestind:bestind + 2]))
        distmat[distmat > cutoffval] = cutoffval
    elif limitDist == 'logAvg':
        cutoffval = np.exp(np.average(np.log(distmat[distmat > 0.0])))
        distmat[distmat > cutoffval] = cutoffval
    else:
        print 'ShRec3D_blobs error: Unknown limitDist, ignoring!'
    # Triangle inequality
    if tri_ineq == 'V2':
        distmat = _enforce_triangleInequalityV2(distmat)
    else:
        distmat = _enforce_triangleInequality(distmat)
    evals, evecs = _distmat2MetricEigs_nD(distmat, 3,
            retAll=False, checkD02=False, ignoreNegD02=False,
            ignoreNegMmat=True, assumeSymm=False)
    evals = np.real(evals)
    coords = np.real(_getEmbedding_3d(evals, evecs))
    return coords, blobrad


def alignCoords_old(coords, coordsr, weights=None):
    """
    Align coords to coordsr.
    """
    if weights is None:
      weights = 1.0
    else:
      weights = np.sqrt(weights)
    data = [0 for i in range(8)]
    datar = [0 for i in range(8)]
    for i in range(2):
      x = coords[:, 0] * (-1) ** i
      for j in range(2):
        y = coords[:, 1] * (-1) ** j
        for k in range(2):
          z = coords[:, 2] * (-1) ** k
          ind = ((i * 2) + j) * 2 + k
          data[ind] = np.array([x, y, z]).T
          datar[ind] = np.array([x * weights,
                                 y * weights,
                                 z * weights]).T
    coordsr2 = coordsr * np.tile(weights, (3, 1)).T
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr2) for c in datar]
    igood = np.argmin(rmsdcheck)
    rotatedscaled = rmsd.kabsch_rotate(datar[igood], coordsr2)
    rotated = rotatedscaled / np.tile(weights, (3, 1)).T
    return rotated


def alignCoords(coords, coordsr, weights=None):
    """
    Align coords to coordsr.
    """
    if weights is None:
      w = np.ones(len(coords))
    else:
      w = np.sqrt(weights)
    data = [coords, -coords]
    datar = [coords * w[:, np.newaxis], -coords * w[:, np.newaxis]]
    coordsr2 = coordsr * w[:, np.newaxis]
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr2) for c in datar]
    igood = np.argmin(rmsdcheck)
    rotatedscaled = rmsd.kabsch_rotate(datar[igood], coordsr2)
    rotated = rotatedscaled / w[:, np.newaxis]
    return rotated


#####################################################

# Reconstruction distance metrics


def DistMatrix(coords, selinds=None):
    """
    Get inter-partition distance matrix.
    If selecting only subset of partitions, set selinds.
    """
    coords2 = coords[selinds] if selinds is not None else coords
    return pdist(coords2, 'euclidean')


def Drms(basedist, thiscoords):
    """
    Compute rms deviation in pairwise inter-partition distances.
    """
    thisdist = DistMatrix(thiscoords)
    return np.sqrt(np.average(np.abs(basedist - thisdist)))


def RMSD(basecoords, thiscoords, selinds=None):
    """
    Compute RMS distance between base and target coordinates.
    """
    coords2 = basecoords[selinds] if selinds is not None else basecoords
    return np.sqrt(np.average(np.sum((coords2 - thiscoords) ** 2, axis=1)))


def PosDist(coordslist):
    """
    Compute distribution of partition positions: mean position and dr.
    """
    coordsarr = np.array(coordslist)
    coordsavg = np.average(coordsarr, axis=0)
    coordsdev = coordsarr - np.tile(coordsavg, (len(coordslist), 1, 1))
    dr = np.sqrt(np.average(np.sum(np.abs(coordsdev) ** 2, axis=2), axis=0))
    return coordsavg, dr

#####################################################

# Dumping structures to PDB files


def _dump_structPDB(fname, coords, clbls, bfactor=None, segment=None):
    """
    Dump reconstruction coordinates / B-factor to PDB file for visualization.
    If bfactor is None, set to 1.00
    Uses PDB format version 3.3
    COLUMNS        DATA  TYPE    FIELD        DEFINITION
-------------------------------------------------------------------------------------
     1 -  6        Record name   "ATOM  "
     7 - 11        Integer       serial       Atom  serial number.
    13 - 16        Atom          name         Atom name.
    17             Character     altLoc       Alternate location indicator.
    18 - 20        Residue name  resName      Residue name.
    22             Character     chainID      Chain identifier.
    23 - 26        Integer       resSeq       Residue sequence number.
    27             AChar         iCode        Code for insertion of residues.
    31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60        Real(6.2)     occupancy    Occupancy.
    61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    67 - 76        Lstring(10)   segment      Segment name.
    77 - 78        LString(2)    element      Element symbol, right-justified.
    79 - 80        LString(2)    charge       Charge  on the atom.
    """
    if bfactor is None:
        bfactor = np.ones_like(coords[:, 0])
    if segment is None:
        segment = ['' for i in range(len(coords))]
    clbl = ''
    ichain = 0
    with open(fname, 'w') as f:
        for iatom, ((x, y, z), bval, cl, seg) in enumerate(zip(coords, bfactor, clbls, segment)):
            if clbl != cl:
                clbl = cl
                ires = 0
                ichain += 1
            ires += 1
            idchain = chr(64 + ichain)
            line = 'ATOM  %5i  CA  GLY %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf%10s C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval, seg)
            #line = 'ATOM  %5i  C5\'  DA %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf           C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval)
            f.write(line)
    pass


def _dump_structPDB_chrs(fname, coords, clbls, bfactor=None, segment=None):
    """
    Dump reconstruction coordinates / B-factor to PDB file for visualization.
    If bfactor is None, set to 1.00
    Uses PDB format version 3.3
    COLUMNS        DATA  TYPE    FIELD        DEFINITION
-------------------------------------------------------------------------------------
     1 -  6        Record name   "ATOM  "
     7 - 11        Integer       serial       Atom  serial number.
    13 - 16        Atom          name         Atom name.
    17             Character     altLoc       Alternate location indicator.
    18 - 20        Residue name  resName      Residue name.
    22             Character     chainID      Chain identifier.
    23 - 26        Integer       resSeq       Residue sequence number.
    27             AChar         iCode        Code for insertion of residues.
    31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60        Real(6.2)     occupancy    Occupancy.
    61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    67 - 76        Lstring(10)   segment      Segment name.
    77 - 78        LString(2)    element      Element symbol, right-justified.
    79 - 80        LString(2)    charge       Charge  on the atom.

    Note: This version maps chr1 -> chain A, chr2 -> chain B, ...,
            chrX -> chain W, chrY -> chain X.
    """
    if bfactor is None:
        bfactor = np.ones_like(coords[:, 0])
    if segment is None:
        segment = ['' for i in range(len(coords))]
    clbl = ''
    cnamelist = [str(i) for i in range(1, 23)] + ['X', 'Y']
    cnamedict = {c: chr(65 + i) for i, c in enumerate(cnamelist)}
    ichain = 0
    with open(fname, 'w') as f:
        for iatom, ((x, y, z), bval, cl, seg) in enumerate(zip(coords, bfactor, clbls, segment)):
            if clbl != cl:
                clbl = cl
                ires = 0
                ichain += 1
            ires += 1
            #idchain = chr(64 + ichain)
            idchain = cnamedict[cl]
            line = 'ATOM  %5i  CA  GLY %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf%10s C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval, seg)
            #line = 'ATOM  %5i  C5\'  DA %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf           C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval)
            f.write(line)
    pass


def _dump_structPDB_multi(fname, coordslist, clblslist, bfactorlist=None,
          segmentlist=None):
    """
    Dump reconstruction coordinates / B-factor to PDB file for visualization.
    Writes multiple models (structures) to the same file.
    If bfactor is None, set to 1.00
    Uses PDB format version 3.3
    COLUMNS        DATA  TYPE    FIELD        DEFINITION
-------------------------------------------------------------------------------------
     1 -  6        Record name   "ATOM  "
     7 - 11        Integer       serial       Atom  serial number.
    13 - 16        Atom          name         Atom name.
    17             Character     altLoc       Alternate location indicator.
    18 - 20        Residue name  resName      Residue name.
    22             Character     chainID      Chain identifier.
    23 - 26        Integer       resSeq       Residue sequence number.
    27             AChar         iCode        Code for insertion of residues.
    31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60        Real(6.2)     occupancy    Occupancy.
    61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    67 - 76        Lstring(10)   segment      Segment name.
    77 - 78        LString(2)    element      Element symbol, right-justified.
    79 - 80        LString(2)    charge       Charge  on the atom.
    """
    with open(fname, 'w') as f:
        for i, (coords, clbls) in enumerate(zip(coordslist, clblslist)):
            f.write('MODEL     %4i\n' % (i + 1))
            if bfactorlist is None:
                bfactor = np.ones_like(coords[:, 0])
            else:
                bfactor = bfactorlist[i]
            if segmentlist is None:
                segment = ['' for i in range(len(coords))]
            else:
                segment = segmentlist[i]
            clbl = ''
            ichain = 0
            for iatom, ((x, y, z), bval, cl, seg) in enumerate(zip(coords, bfactor, clbls, segment)):
                if clbl != cl:
                    clbl = cl
                    ires = 0
                    ichain += 1
                ires += 1
                idchain = chr(64 + ichain)
                line = 'ATOM  %5i  CA  GLY %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf%10s C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval, seg)
                #line = 'ATOM  %5i  C5\'  DA %s%4i    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf           C\n' % (iatom, idchain, ires, x, y, z, 1.0, bval)
                f.write(line)
            f.write('ENDMDL\n')
    pass


#####################################################

# PyMol visualizations


def PyMolSnapshot(coords, clbls, fname, bfactor=None):
    """
    Create pymol snapshot of structure, spheres colored by chain.
    """
    pymol.finish_launching()
    fname2 = 'temp.pdb'
    _dump_structPDB(fname2, coords, clbls, bfactor=bfactor)
    cmd.load(fname2)
    util.cbc()
    cmd.show_as(representation='spheres')
    cmd.mpng(fname, 1, 1)
    sleep(0.5)
    cmd.delete('all')


def JitterMovie(basecoords, coordslist, clbls, fname, selindslist=None,
                factor=1.0):
    """
    Create GIF movie of partition positions using PyMol
    """
    print 'Creating snapshots...'
    PyMolSnapshot(basecoords * factor, clbls, 'base.png', bfactor=None)
    if selindslist is None:
        selindslist = [np.arange(len(clbls))] * len(coordslist)
    for i, (thiscoords, thisselinds) in enumerate(zip(coordslist, selindslist)):
        thisclbls = np.array(clbls)[thisselinds]
        PyMolSnapshot(thiscoords * factor, thisclbls, '%03i-temp.png' % i,
                        bfactor=None)
    print 'Done. Converting snapshots to gif...'
    cmdlist = ['convert', '-delay', '50', '-loop', '0',
                'base*.png', '*-temp*.png', fname]
    call(cmdlist)
    print 'Done. Cleaning up...'
    cmdlist = ' '.join(['rm', 'base*.png', '*-temp*.png'])
    call(cmdlist, shell=True)


def JitterPlot(coords, dr, clbls, fname, factor=1.0, maxdr=None):
    """
    Create "temperature" plot showing position uncertainty as colors.
    """
    pymol.finish_launching()
    fname2 = 'temp.pdb'
    _dump_structPDB(fname2, coords * factor, clbls, bfactor=dr)
    cmd.load(fname2)
    cmd.show_as(representation='spheres')
    if maxdr is None:
        cmd.spectrum('b', 'blue_white_red')
    else:
        cmd.spectrum('b', 'blue_white_red', 'all', 0.0, maxdr)
    cmd.mpng(fname, 1, 1)
    sleep(0.5)
    cmd.delete('all')

