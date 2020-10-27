
import numpy as np
import sys
from scipy.stats import linregress
import plotutils as plu
import matplotlib.pyplot as plt


# Interaction matrix normalization routines


def norm_psize0(fij, si):
  """
  f'ij = fij / (si * sj)
  """
  return fij / np.outer(si, si)


def norm_SCN0(fij, delta=1.0e-5, maxiter=1000):
  """
  f'ij = fij / |fi|  (Euclidean norm)
  f*ij = f''ij = f'ij / |fj|
  Iterate to convergence
  """
  dlist = []
  convergedsteps = 10
  nparts = len(fij)
  f = fij.copy()
  for i in range(maxiter):
    rownorm = np.sqrt(np.sum(f ** 2, axis=1))
    f1 = f / rownorm[:, None]
    colnorm = np.sqrt(np.sum(f1 ** 2, axis=0))
    f2 = f1 / colnorm[None, :]
    d = np.sum(np.abs(f2 - f2.T)) / np.sum(np.abs(f2))
    if d < delta:
      return (f2 + f2.T) / 2.0
    else:
      f = f2
      # Check if d has been constant for convergedsteps iterations
      if len(dlist) > convergedsteps and \
          np.allclose(dlist[-convergedsteps:], d, rtol=5.0e-3):
        print 'SCN stuck at d = %.2e .' % d
        return (f2 + f2.T) / 2.0
      dlist.append(d)
  print 'SCN Not converged after %i iterations, d = %.2e .' % (maxiter, d)
  return (f2 + f2.T) / 2.0


def norm_rownorm10(fij):
  """
  f'ij = fij / |fi||fj|  (Euclidean norm)
  """
  rownorm = np.sqrt(np.sum(fij ** 2, axis=1))
  return fij / np.outer(rownorm, rownorm)


def norm_rowsum10(fij):
  """
  f'ij = fij / (fi * fj)  (Row sums)
  """
  rowsum = np.sum(fij, axis=1)
  return fij / np.outer(rowsum, rowsum)


def norm_rownormIter0(fij, delta=1.0e-5, maxiter=1000):
  """
  f'ij = fij / |fi||fj|  (Euclidean norm)
  Iterate until matrix power of antisymmetric part is less than delta of
    full matrix.
  """
  f = fij.copy()
  for i in range(maxiter):
    f2 = norm_rownorm1(f)
    d = np.sum(np.abs(f2 - f2.T)) / np.sum(np.abs(f2))
    if d < delta:
      return f2
    else:
      f = f2
  print 'Not converged after %i iterations, d = %.2e .' % (maxiter, d)
  return f2


def norm_rowsumIter0(fij, delta=1.0e-5, maxiter=1000):
  """
  f'ij = fij / (fi * fj)  (Row sums)
  Iterate until matrix power of antisymmetric part is less than delta of
    full matrix.
  """
  f = fij.copy()
  for i in range(maxiter):
    f2 = norm_rowsum1(f)
    d = np.sum(np.abs(f2 - f2.T)) / np.sum(np.abs(f2))
    if d < delta:
      return f2
    else:
      f = f2
  print 'Not converged after %i iterations, d = %.2e .' % (maxiter, d)
  return f2


def fmatRescale_ratio(fmat, mappingdata, dist, avgVsDist):
    mapping, nbins = mappingdata
    avgCount = np.sum(fmat) / (2 * len(fmat) * (len(fmat) - 1))
    fmatpad = plu._build_fullarray(fmat, mappingdata, 0.0)
    factormat = np.eye(len(fmatpad))
    for  d, r in zip(dist, avgVsDist):
        val = r
        np.fill_diagonal(factormat[d:], val)
        np.fill_diagonal(factormat[:, d:], val)
#     for d, r in zip(dist, avgVsDist):
#         if np.isnan(r) or r == 0.0:
#             continue
#         m = np.diag([r] * (nbins - d), k=d)
#         factormat += m
#     factormat += factormat.T
    fmatpad2 = fmatpad / factormat
    return fmatpad2[mapping][:, mapping] * avgCount

def preprocess_fmat(fmat, mappingdata, plot=True):
    """
    Output: allDist, allFij, dist, avgVsDist, nAtDist
    """
    fmatpad = plu._build_fullarray(fmat, mappingdata, 0.0)
    mapping, nbins = mappingdata
    if plot:
        f, x = plt.subplots(1, 2, figsize=(14, 6))
    allDist = []
    allFij = []
    avgVsDist = []
    nAtDist = []
    for d in range(1, nbins):
        v = np.diag(fmatpad, k=d)
        v = v[v > 0.0]
        avgVsDist.append(np.average(v))
        nAtDist.append(np.sum(v > 0.0))
        if plot:
            _ = x[0].scatter([d] * len(v), v, s=1, color='b')
        allDist.extend([d] * len(v))
        allFij.extend(v)
    if plot:
        _ = x[1].plot(range(1, nbins), avgVsDist)
        _ = x[0].set_xscale('log')
        _ = x[0].set_yscale('log')
        _ = x[0].set_ylim(ymin=1.0)
        _ = x[0].set_title('$F(s)$')
        _ = x[0].set_xlabel('Genomic distance $s = |i-j|$')
        _ = x[0].set_ylabel('Interaction strength $F_{ij}$')
        _ = x[1].set_xscale('log')
        _ = x[1].set_yscale('log')
        _ = x[1].set_ylim(ymin=1.0)
        _ = x[1].set_title('$\\bar{F}(s)$')
        _ = x[1].set_xlabel('Genomic distance $s = |i-j|$')
        _ = x[1].set_ylabel('Average interaction strength $\\bar{F}_{ij}$')
    return np.array(allDist), np.array(allFij), np.arange(1, nbins), np.array(avgVsDist), nAtDist


def fmatRescale_powlaw(fmat, mappingdata, allDist, allFij, dist, nb=100, plot=True):
    mapping, nbins = mappingdata
    avgCount = np.sum(fmat) / (2 * len(fmat) * (len(fmat) - 1))
    fmatpad = plu._build_fullarray(fmat, mappingdata, 0.0)
    ## Power law fit
    minDist = np.min(allDist)
    maxDist = np.max(allDist)
    binsDist = np.logspace(np.log10(minDist), np.log10(maxDist), nb + 1)
    binctr = []
    binval = []
    for i in range(nb):
        st, en = binsDist[i:i + 2]
        mask = (allDist >= st) & (allDist < en)
        v = allFij[mask]
        if len(v) == 0:
            continue
        else:
            binval.append(np.average(v))
            binctr.append(np.sqrt(st * en))
    x = np.log(binctr)
    y = np.log(binval)
    c1, c0, rv, pv, er = linregress(x, y)
#     c1, c0 = np.polyfit(x, y, 1)
    xfit = np.exp(x)
    xfit = dist
    yfit = np.exp(c0) * xfit ** c1
    if plot:
        f, x = plt.subplots(1, 1)
        _ = x.plot(xfit, yfit, label='fit')
        _ = x.plot(binctr, binval, label='average')
        _ = x.set_xlabel('Genomic distance')
        _ = x.set_ylabel('Average interaction strength')
        _ = x.set_yscale('log')
        _ = x.set_xscale('log')
        _ = plt.legend()
    factormat = np.eye(len(fmatpad))
    for d, (x, r) in enumerate(zip(xfit, yfit)):
        val = r
        np.fill_diagonal(factormat[x:], val)
        np.fill_diagonal(factormat[:, x:], val)
#     for d, (x, r) in enumerate(zip(xfit, yfit)):
#         m = np.diag([r] * (nbins - x), k=x)
#         m += m.T
#         factormat += m
    fmatpad3 = fmatpad / factormat
    return fmatpad3[mapping][:, mapping] * avgCount


def genomeFab_intraScale_ratio(fab, mappingdata, chainpadarray, cnamelist):
    """
    Scale intra-chr interactions by interaction ratios as a function of genomic distance.
    """
    fmatpad = plu._build_fullarray(fab, mappingdata, 0.0)
    for c in cnamelist:
        mask = (chainpadarray == c[3:])
        thisfmat = fmatpad[mask][:, mask]
        chrshift = np.min(np.nonzero(mask)[0])
        thismappingdata = np.array([i for i in np.nonzero(mask)[0] if i in mappingdata[0]]) - chrshift, np.sum(mask)
        thisfmat2 = thisfmat[thismappingdata[0]][:, thismappingdata[0]]
        allDist, allFij, dist, avgVsDist, nAtDist = preprocess_fmat(thisfmat2, thismappingdata, plot=False)
        thisfmatratio = fmatRescale_ratio(thisfmat2, thismappingdata, dist, avgVsDist)
        thischrinds = np.sort(np.nonzero(mask)[0])[thismappingdata[0]]
        for i, j in enumerate(thischrinds):
            fmatpad[j, thischrinds] = thisfmatratio[i]
    return fmatpad[mappingdata[0]][:, mappingdata[0]]


def genomeFab_intraScale_pow(fab, mappingdata, chainpadarray, cnamelist):
    """
    Scale intra-chr interactions by interaction ratios as a function of genomic distance.
    """
    fmatpad = plu._build_fullarray(fab, mappingdata, 0.0)
    for c in cnamelist:
        mask = (chainpadarray == c[3:])
        thisfmat = fmatpad[mask][:, mask]
        chrshift = np.min(np.nonzero(mask)[0])
        thismappingdata = np.array([i for i in np.nonzero(mask)[0] if i in mappingdata[0]]) - chrshift, np.sum(mask)
        thisfmat2 = thisfmat[thismappingdata[0]][:, thismappingdata[0]]
        allDist, allFij, dist, avgVsDist, nAtDist = preprocess_fmat(thisfmat2, thismappingdata, plot=False)
        thisfmatpow = fmatRescale_powlaw(thisfmat2, thismappingdata, allDist, allFij, dist, nb=100, plot=False)
        thischrinds = np.sort(np.nonzero(mask)[0])[thismappingdata[0]]
        for i, j in enumerate(thischrinds):
            fmatpad[j, thischrinds] = thisfmatpow[i]
    return fmatpad[mappingdata[0]][:, mappingdata[0]]


def norm_SCPN_old(fij, weights, delta=1.0e-10, maxiter=1000, ntype='norm'):
  """
  f'ij = fij / |fi|  (Euclidean norm)
  f*ij = f''ij = f'ij / |fj|
  Iterate to convergence
  """
  nparts = len(fij)
  f = fij.copy()
  for i in range(maxiter):
      if ntype == 'norm':
          rownorm = np.sqrt(np.sum(f ** 2, axis=1))
      elif ntype == 'sum':
          rownorm = np.sum(f, axis=1)
      r = np.tile(rownorm / weights, (nparts, 1)).T
      f1 = f / r
      if ntype == 'norm':
          colnorm = np.sqrt(np.sum(f1 ** 2, axis=0))
      elif ntype == 'sum':
          colnorm = np.sum(f1, axis=0)
      r = np.tile(colnorm / weights, (nparts, 1))
      f2 = f1 / r
      d = np.sum(np.abs(f2 - f2.T)) / np.sum(np.abs(f2))
      if d < delta:
          return (f2 + f2.T) / 2.0 / np.outer(weights, weights)
      else:
          f = f2
  print 'SCPN Not converged after %i iterations, d = %.2e .' % (maxiter, d)
  return (f2 + f2.T) / 2.0 / np.outer(weights, weights)


def norm_SCPN(fij, weights, delta=1.0e-10, maxiter=1000, ntype='norm'):
  """
  f'ij = fij / |fi|  (Euclidean norm)
  f*ij = f''ij = f'ij / |fj|
  Iterate to convergence
  """
  f = fij.copy()
  for i in range(maxiter):
      if ntype == 'norm':
          rownorm = np.sqrt(np.sum(f ** 2, axis=1))
      elif ntype == 'sum':
          rownorm = np.sum(f, axis=1)
      f1 = f / (rownorm / weights)[:, np.newaxis]
      if ntype == 'norm':
          colnorm = np.sqrt(np.sum(f1 ** 2, axis=0))
      elif ntype == 'sum':
          colnorm = np.sum(f1, axis=0)
      f2 = f1 / (colnorm / weights)[np.newaxis, :]
      d = np.sum(np.abs(f2 - f2.T)) / np.sum(np.abs(f2))
      if d < delta:
          return (f2 + f2.T) / 2.0 / np.outer(weights, weights)
      else:
          f = f2
  print 'SCPN Not converged after %i iterations, d = %.2e .' % (maxiter, d)
  return (f2 + f2.T) / 2.0 / np.outer(weights, weights)

def norm_raw(fab, **kwargs):
  return fab.copy()

def norm_psize(fab, **kwargs):
  if 'psizes' in kwargs:
    psizes = kwargs['psizes']
    return norm_psize0(fab, psizes)
  else:
    return fab.copy()

def norm_SCN(fab, **kwargs):
  return norm_SCN0(fab)

def norm_SCPNn(fab, **kwargs):
  return norm_SCPN(fab, kwargs['psizes'], ntype='norm')

def norm_SCPNs(fab, **kwargs):
  return norm_SCPN(fab, kwargs['psizes'], ntype='sum')

def norm_rownorm1(fab, **kwargs):
  return norm_rownorm10(fab)

def norm_rownormIter(fab, **kwargs):
  return norm_rownormIter0(fab)

def norm_rowsum1(fab, **kwargs):
  return norm_rowsum10(fab)

def norm_rowsumIter(fab, **kwargs):
  return norm_rowsumIter0(fab)

def fab2dab(fab, **kwargs):
    alpha = kwargs.get('alpha', sys.float_info.epsilon)
    beta = kwargs.get('beta', 1.0)
    gamma = kwargs.get('gamma', 1.0)
    dab = (fab ** beta + alpha) ** (-gamma)
    np.fill_diagonal(dab, 0.0)
    #dab /= np.sum(dab)
    return dab

