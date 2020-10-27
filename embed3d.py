
import numpy as np
import rec3Dutils as rtd
import coordcomp as cc
import time


# Functions for reconstruction using 3DEmbed


def distConstraint(d, limitDist):
    """
    Implement constraints to limit maximum distances
    """
    if limitDist == 'None':
        pass
    elif limitDist == 'logMax':
        hist, edges = np.histogram(np.log(d[d > 0.0]), bins=100)
        bestind = np.argmax(hist)
        cutoffval = np.exp(np.average(edges[bestind:bestind + 2]))
        d[d > cutoffval] = cutoffval
    elif limitDist == 'logAvg':
        cutoffval = np.exp(np.average(np.log(d[d > 0.0])))
        d[d > cutoffval] = cutoffval
    else:
        print 'ShRec3D_blobs error: Unknown limitDist, ignoring!'
    return d


def distTriIneq(d, tri_ineq):
    """
    Choose function to implement triangle inequality on distance matrix
    """
    if tri_ineq == 'V2':
        return rtd._enforce_triangleInequalityV2(d)
    else:
        return rtd._enforce_triangleInequality(d)



def dist2Recon(Dij, **kwargs):
    """
    Given distance matrix, reconstruct 3D embedding using
      distance geometry.
    Input:
      Dij: (N, N) distance matrix
    Given kwargs:
      limitDist: kwargs['limitDist'], choose how to set
                   a cutoff for Dij (default: 'None')
      tri_ineq: kwargs['tri_ineq'], choose how to enforce
                  triangle inequality (default: 'V2')
      sizeScale: kwargs['sizeScale'], rescale coords to this
                   average distance from origin (default: 20.0)
    Returns: coords, evals
      coords: (N, 3) array of XYZ coordinates
      evals: (N,) array of metric eigenvalues
    """
    limitDist = kwargs.get('limitDist', 'None')
    tri_ineq = kwargs.get('tri_ineq', 'V2')
    sizeScale = kwargs.get('sizeScale', 20.0)
    takePosEvals = kwargs.get('takePosEvals', False)
    # Implement Dij constraints
    Dij = distConstraint(Dij, limitDist)
    # Triangle inequality
    Dij = distTriIneq(Dij, tri_ineq)
    # Reconstruct
    evals, evecs = rtd._distmat2MetricEigs_nD(Dij, 3,
            retAll=True, checkD02=False, ignoreNegD02=True,
            ignoreNegMmat=True, assumeSymm=True, takePosEvals=takePosEvals)
    evals = np.real(evals)
    coords = np.real(rtd._getEmbedding_3d(evals[:3], evecs[:, :3]))
    # Center and rescale
    coords = cc.centerRescaleCoords(coords, sizeScale)
    return coords, evals


