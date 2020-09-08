
import numpy as np
import rmsd


# Functions for coordinate manipulation and computations


def centerRescaleCoords(coords, scale, weights=None):
    """
    Center coords at origin (assuming each point has equal mass), then
    rescale coordinates such that average distance from origin equals scale.
    If weights is nor None, apply weighting to centroid computation,
    and not to rescaling.
    """
    if weights is None:
      weights = np.ones(len(coords))
    coords2 = coords.copy()
    xav = np.sum(coords2[:, 0] * weights) / np.sum(weights)
    yav = np.sum(coords2[:, 1] * weights) / np.sum(weights)
    zav = np.sum(coords2[:, 2] * weights) / np.sum(weights)
    coords2[:, 0] -= xav
    coords2[:, 1] -= yav
    coords2[:, 2] -= zav
    if scale is not None:
      rms = np.average(np.sqrt(np.sum((coords2) ** 2, axis=1)))
      coords2 *= scale / rms
    return coords2


def constraintMaxRadius(coords, maxrad, trunc=None):
    """
    If any point is beyond maxrad from the origin, use function trunc to truncate
    radius.
    If trunc is None, set to hard sphere cutoff
    """
    c = coords.copy()
    if trunc is None:
      trunc = lambda x: 0.0
    rads = np.sqrt(np.sum(c ** 2.0, axis=1))
    for i, r in enumerate(rads):
      if r > maxrad:
        rv = maxrad + trunc(r - maxrad)
        c[i] *= rv / r
    return c


def coords2Dist(coords):
  """
  Coordinate matrix to distance matrix
  """
  npts = len(coords)
  dx = np.tile(coords[:, 0], (npts, 1))
  dx = dx - dx.T
  dy = np.tile(coords[:, 1], (npts, 1))
  dy = dy - dy.T
  dz = np.tile(coords[:, 2], (npts, 1))
  dz = dz - dz.T
  d = np.sqrt(dx ** 2.0 + dy ** 2.0 + dz ** 2.0)
  return d


def RMSD(coords1, coords2, weights=None):
  """
  RMSD between two structures (assumed to be aligned already)
  """
  if weights is None:
    weights = np.ones(len(coords1))
  dx2 = (coords1 - coords2)[:, 0] ** 2.0
  dy2 = (coords1 - coords2)[:, 1] ** 2.0
  dz2 = (coords1 - coords2)[:, 2] ** 2.0
  return np.sqrt(np.sum((dx2 + dy2 + dz2) * weights) / np.sum(weights))
  #return np.sqrt(np.average(np.sum((coords1 - coords2) ** 2.0, axis=1)))


def dRMS(coords1, coords2, weights=None):
  """
  dRMS between two structures
  """
  if weights is None:
    weights = np.ones(len(coords1))
  wmat = np.outer(weights, weights)
  d1 = coords2Dist(coords1)
  d2 = coords2Dist(coords2)
  return np.sqrt(np.sum((d1 - d2) ** 2.0 * wmat) / np.sum(wmat))


def dRMS_refDist(coords, refdist):
  """
  dRMS between two structures
  """
  d1 = coords2Dist(coords)
  return np.sqrt(np.average((d1 - refdist) ** 2.0))


def alignCoords_mixRes(coords, coordsr, mapping, mappingr, weights=None, weightsr=None):
    """
    Align coords to coordsr.
    """
    inds = list(set(list(np.unique(mapping)) + list(np.unique(mappingr))))
    if weights is None:
      weights = np.ones(len(mapping))
    else:
      weights = np.sqrt(weights)
    if weightsr is None:
      weightsr = np.ones(len(mappingr))
    else:
      weightsr = np.sqrt(weightsr)
    c1 = np.array([np.average(coords[mapping == ind], axis=0) for ind in inds])
    c2 = np.array([np.average(coordsr[mappingr == ind], axis=0) for ind in inds])
    w1 = np.array([np.sum(weights[mapping == ind]) for ind in inds])
    w2 = np.array([np.sum(weightsr[mappingr == ind]) for ind in inds])
    wmask = (w1 > 0) & (w2 > 0)
    data = [0 for i in range(8)]
    datac = [0 for i in range(8)]
    datar = [0 for i in range(8)]
    for i in range(2):
      x = c1[:, 0] * (-1) ** i
      xc = coords[:, 0] * (-1) ** i
      for j in range(2):
        y = c1[:, 1] * (-1) ** j
        yc = coords[:, 1] * (-1) ** j
        for k in range(2):
          z = c1[:, 2] * (-1) ** k
          zc = coords[:, 2] * (-1) ** k
          ind = ((i * 2) + j) * 2 + k
          data[ind] = np.array([x, y, z]).T
          datac[ind] = np.array([xc, yc, zc]).T
          datar[ind] = (np.array([x * w1, y * w1, z * w1]).T)[wmask]
    coordsr2 = (c2 * np.tile(w2, (3, 1)).T)[wmask]
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr2) for c in datar]
    igood = np.argmin(rmsdcheck)
    rotmat = rmsd.kabsch(datar[igood], coordsr2)
    rotated = np.dot(datac[igood], rotmat)
    #rotatedscaled = rmsd.kabsch_rotate(datar[igood], coordsr2)
    #rotated = rotatedscaled / np.tile(weights, (3, 1)).T
    return rotated


