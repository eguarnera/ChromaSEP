
"""
multiHierarchy.py: Functions for performing, characterizing, and testing
    subpartitioning of low-level partitions.

Naming conventions:
  - We assume that L0 partitions are obtained from whole-chromosome
    partitioning
  - L1 partitions are obtained by partitioning individual L0 partitions

"""


import numpy as np
import sys
import os
import cPickle as pickle
import copy
import itertools

import hicutils as hcu
import plotutils as plu
from ChromaWalker import ChromaWalker
import msmTPT as mt
import msmTsetOpt as mto


#####################################

# Loading full-chromosome and L0 arrays


def getData_basecase(cw, betamap, cname, rhomax):
  """
  Retrieve basic full-chromosome data and L0-TsetOpt results.

  Returns: (beta, arrays, dicts, nmins)
    - beta:   Annealing parameter used in this case
    - arrays: (fmat, mmat, cmat, mappingdata)
    - dicts:  (rdict, tdict)
    - nmins:  Minima of rho
  """
  beta = betamap[cname]
  fmat, mmat, cmat, mappingdata = cw.DFR.get_arrays(cname, beta)
  rdict, tdict = cw.DFR.get_datadicts(cname)
  nmins = findMinima(rdict, beta, rhomax)
  return beta, (fmat, mmat, cmat, mappingdata), (rdict, tdict), nmins


def getData_basecaseLevelpartitions(cw, cname, beta, mappingdata, tdict,
                nmins, minind):
  """
  Retrieve L0 partition data.

  Returns: (thistset, thism, thisn)
    - thistset: Target hub set
    - thism:    Membership array (hard partitioning)
    - thisn:    Number of hubs defining L0 partitions
  """
  thisn = nmins[minind]
  thism, _, _ = cw.TOpt.get_partitions(cname, beta, thisn)
  thistset = tdict[beta, thisn]
  thism = thism[:, mappingdata[0]]
  thism = thism[np.sum(thism, axis=1) > 0]
  return thistset, thism, thisn


def getData_nextcaseLeveldata(datadir, cname, minind, fmat, thism, nmax=30):
  """
  Get L0 subarray data.
  Input:
    - datadir: Archive directory
    - cname:   Chromosome name
    - minind:  Select minind-th minimum in full-chromosome rho
    - fmat:    Full-chromosome fmat
    - thism:   Selected hard membership array
    - nmax:    Max number of L1 partitions to find in each L0 partition
               (default: 30)

  Returns: (fmats, mmats, cmats, nmaxs, lens)
    - fmats: List of fmat for each L0 partition
    - mmats: List of mmat for each L0 partition
    - cmats: List of cmat for each L0 partition
    - nmaxs: List of nmax for each L0 partition
    - lens:  List of sizes of each L0 partition
  """
  fmats = [fmat[(mvec>0)][:, (mvec>0)].copy() for mvec in thism]
  mmats = ['' for fm in fmats]
  for ipart, fm in enumerate(fmats):
    fname = os.path.join(datadir, 'mats', 'mmat-%s-%02i-%02i.dat' % (cname, minind, ipart))
    if os.path.isfile(fname):
      m = np.fromfile(fname, 'float64')
      nbins = int(np.sqrt(len(m)))
      m.shape = nbins, nbins
      mmats[ipart] = m
    else:
      mmats[ipart] = hcu._calc_MFPT(fm)
      fname = os.path.join(datadir, 'mats', 'mmat-%s-%02i-%02i.dat' % (cname, minind, ipart))
      mmats[ipart].tofile(fname)
  cmats = [hcu._calc_cmat(mm) for mm in mmats]
  lens = [len(fm) for fm in fmats]
  nmaxs = [min(nmax, l / 4) for l in lens]
  return fmats, mmats, cmats, nmaxs, lens


#####################################

# TsetOpt on L0 partitions


def refineTsets(c, rhodata, tsetdata, ntargetmax, beta, steppars=None):
  """
  Equivalent of ConstructMC on a single L0 partition, without saving
    rhodata / tsetdata to file.

  Returns:
    - nupdate: Number of entries updated
  """
  if steppars is None:
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = \
            100, 0.5, 200, 0.1, 1.0
    steppars = nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT
  gammamaxmap = np.array([np.argmax(v) for v in c])
  allchoices = range(len(c))
  rhofunc = mt._rhoindex
  targetset = copy.deepcopy(tsetdata[beta, 2])
  rtemp = {}
  ttemp = {}
  for i in range(2, ntargetmax):
    ntarget = i + 1
    ### Optimize new target
    newtarget, newrho = mto._trynewtarget_construct(c, targetset,
                rhofunc=rhofunc)
    targetset.append(newtarget)
    if len(targetset) != ntarget:
      print 'Error! ntarget mismatch 1!'
      sys.exit()
    bestrho, besttset = mto._ConstructTset_MCn(steppars, newrho,
                targetset, allchoices, c, gammamaxmap,
                rhofunc=rhofunc)
    ### Record rhodata, targetsetdata
    targetset = besttset
    if len(targetset) != ntarget:
      print 'Error! ntarget mismatch 2!'
      sys.exit()
    key = (beta, ntarget)
    rtemp[key] = bestrho
    ttemp[key] = copy.deepcopy(targetset)
  #     print i + 1, ':', bestrho
  nupdate = 0
  for k in rtemp:
      if k not in rhodata or rhodata[k] > rtemp[k]:
          print ('Add' if (k not in rhodata) else 'Update'), k
          rhodata[k] = rtemp[k]
          tsetdata[k] = copy.deepcopy(ttemp[k])
          nupdate += 1
  return nupdate


def OptimizeTsets(rdicts, tdicts, fmats, mmats, cmats, nmaxs, nvalss, beta,
                  tsetOptPars):
  """
  Equivalent of iterated (ConstructMC + PertSnap) for each L0 partition
    defined from a full L0 hub set, except:
      - No data saved to file
      - PertSnap replaced by random sampling within hard partitions
        obtained from current optimal tset

  Returns:
    - qhardss: List of hard-partitioning membership functions
  """
  nruns, nreps, ntrials, minupdate = tsetOptPars
  qhardss = ['' for nmax in nmaxs]
  for ipart, (fm, mm, cm, nmax, nv) in enumerate(
            zip(fmats, mmats, cmats, nmaxs, nvalss)):
    rd = rdicts[ipart]
    td = tdicts[ipart]
    for rep in range(nreps):
      print '**************************************'
      print 'TsetOpt partition', ipart, 'Rep', rep
      for i in range(nruns):
        nupdate = refineTsets(cm, rd, td, nmax, beta)
        if nupdate < minupdate:
          break
      qs = []
      for n in nv:
        tset = list(np.sort(td[beta, n]))
        qAip = mt._calc_qAi_exact(fm, tset)
        qAi = np.array([qAip[t] for t in np.sort(qAip.keys())])
        qs.append(qAi)
      qhards = []
      for qAi in qs:
        q = np.zeros_like(qAi)
        for i in range(len(qAi.T)):
          q[np.argmax(qAi[:, i]), i] = 1.0
        qhards.append(q)
      print '**************************************'
      print 'Sample partition'
      nupdate = 0
      for i, n in enumerate(nv):
        # Get current tset
        tset_cur = np.sort(td[beta, n])
        # Get current partitions
        m_cur = qhards[i]
        choices = [np.nonzero(m)[0] for m in m_cur]
        # Perturb tset using partitions
        changed = False
        for itrial in range(ntrials):
          tset_test = copy.deepcopy(tset_cur)
          ## Option 1: Randomize one partition choice
          ind = np.random.randint(len(choices))
          tset_test[ind] = np.random.choice(choices[ind])
          # Test trial tset
          trialrho = mt._rhoindex(cm, tset_test)
          if trialrho < rd[beta, n]:
            changed = True
            print 'Improved rhoA[', beta, n, ']: %.2e' % trialrho
            rd[beta, n] = trialrho
            td[beta, n] = copy.deepcopy(tset_test)
        # If updated tset, update qAs and qhardAs
        if changed:
          tset = list(np.sort(td[beta, n]))
          qAip = mt._calc_qAi_exact(fm, tset)
          qAi = np.array([qAip[t] for t in np.sort(qAip.keys())])
          qs[i] = qAi
          q = np.zeros_like(qAi)
          for j in range(len(qAi.T)):
              q[np.argmax(qAi[:, j]), j] = 1.0
          qhards[i] = q
          nupdate += 1
      print '**************************************'
      print 'Check that hubs remain in partition'
      for i, n in enumerate(nv):
        # Get current tset
        tset_cur = np.sort(td[beta, n])
        # Get current partitions
        m_cur = qhards[i]
        choices = [np.nonzero(m)[0] for m in m_cur]
        for t, c in zip(tset_cur, choices):
          if t not in c:
            print i, n, t
      qhardss[ipart] = copy.deepcopy(qhards)
      if nupdate < minupdate:
        break
  return qhardss


def getData_nextcaseTsetOpt_load(datadir, cname, beta, minind, rhomax):
  """
  Load TsetOpt results on L0 partitions
  """
  fname = os.path.join(datadir, 'dicts', 'datadicts-%s-%02i.p' % (cname, minind))
  rdicts, tdicts = pickle.load(open(fname, 'rb'))
  nminss = [findMinima(rd, beta, rhomax) for rd in rdicts]
  return rdicts, tdicts, nminss


#####################################

# Computing / storing / retrieving L1 partition data


def getData_qAi(fmat, tset):
  """
  Basic computation of qAi given interaction fmat and target hub set tset.
  """
  t = list(np.sort(tset))
  qAip = mt._calc_qAi_exact(fmat, t)
  qAi = np.array([qAip[t] for t in np.sort(qAip.keys())])
  return qAi


def getData_nextlevelQAi(cw, betamap, rhomax, datadir, cname, minind, ipart, nn, thislen):
  """
  Compute committor function qAi for L1 partitions, and archive in datadir.
  If archived data available, read from file.
  Input:
    - datadir: Archive directory
    - cname:   Chromosome name
    - minind:  L0 partitions defined by minind-th minimum
               in full-chromosome rho to define
    - ipart:   Select ipart-th L0 partition
    - nn:      Partition selected L0 partition into nn L1 partitions
    - thislen: Size of selected L0 partition

  Returns:
    - qAi:     Committor function (soft partitioning)
  """
  if nn == 1:
    return np.ones((1, thislen))
  thisdir = os.path.join(datadir, 'qAis', cname, 'minind%02i' % minind, 'ipart%02i' % ipart)
  if not os.path.isdir(thisdir):
    os.makedirs(thisdir)
  fname = os.path.join(thisdir, 'qAi-nn%02i.dat' % nn)
  if os.path.isfile(fname):
    # Load data
    qAi = np.fromfile(fname, 'float64')
    qAi.shape = len(qAi) / thislen, thislen
    return qAi
  else:
    beta, (fmat, mmat, cmat, mappingdata), (rdict, tdict), nmins = getData_basecase(cw, betamap, cname, rhomax)
    thistset, thism, thisn = getData_basecaseLevelpartitions(cw, cname, beta, mappingdata, tdict, nmins, minind)
    fmats, mmats, cmats, nmaxs, lens = getData_nextcaseLeveldata(datadir, cname, minind, fmat, thism)
    rdicts, tdicts, nminss = getData_nextcaseTsetOpt_load(datadir, cname, beta, minind, rhomax)
    # Create data
    td = tdicts[ipart]
    f = fmats[ipart]
    tset = list(np.sort(td[beta, nn]))
    qAi = getData_qAi(f, tset)
    qAi.tofile(fname)
    print 'Case %s %02i %02i %02i created!' % (cname, minind, ipart, nn)
    return qAi


def soft2hardMembership(qAi):
  """
  Convert soft-partition membership function to hard-partitions.
  """
  q = np.zeros_like(qAi)
  for i in range(len(qAi.T)):
    q[np.argmax(qAi[:, i]), i] = 1.0
  return q


def buildCombinedMembership(thisqs, lens):
  """
  Build combined membership array, and other utility arrays
  """
  nL0 = len(lens)
  nparts = [len(q) for q in thisqs]
  cumnparts = [0] + list(np.cumsum(nparts))
  cumlens = [0] + list(np.cumsum(lens))
  predm = np.zeros((np.sum(nparts), np.sum(lens)))
  for ipart in range(nL0):
    stp, enp = cumnparts[ipart], cumnparts[ipart + 1]
    sti, eni = cumlens[ipart], cumlens[ipart + 1]
    for i, ip in enumerate(range(stp, enp)):
      predm[ip, sti:eni] = thisqs[ipart][i]
  return predm, nparts, cumnparts, cumlens


def buildCombinedTset(beta, cmats, lens, tdicts, ns,
        predm, cumlens, cumnparts, thistset):
  """
  Build combined tset. L0 partitions represented by singleton hub sets use
    original hubs if it is present in the hard partition, otherwise use
    a hub that maximizes gammain.
  """
  tsets = []
  for ipart, nn in enumerate(ns):
    if nn == 1:
      ## Does the partition contain an original tset hub?
      mvec = predm[cumnparts[ipart]]
      t = locateOriginalHub(mvec, thistset)
      if t is None:
        c = cmats[ipart]
        t = np.argmax(np.min(c + np.eye(len(c)), axis=0))   # Maximize gin
      tsets.append(np.array([t]))
    else:
      tsets.append(np.array(tdicts[ipart][beta, nn]))
  trial_tset = []
  for ipart, t in enumerate(tsets):
    trial_tset += list(t + cumlens[ipart])
  trial_tset = np.array(trial_tset)
  return trial_tset, tsets


def buildCombinedTsetPartition(cw, betamap, rhomax, datadir, cname, minind, beta, dictss, leveldata,
        ns, thistset):
  rdicts, tdicts = dictss
  fmats, mmats, cmats, nmaxs, lens = leveldata
  # Retrieve L1 hard partitioning
  thisqs = [soft2hardMembership(
              getData_nextlevelQAi(cw, betamap, rhomax, datadir, cname, minind, ipart, nn, thislen))
            for ipart, (nn, thislen) in enumerate(zip(ns, lens))]
  # Build combined membership
  predm, nparts, cumnparts, cumlens = buildCombinedMembership(thisqs, lens)
  # Get tsets and combined tset
  trial_tset, tsets = buildCombinedTset(beta, cmats, lens, tdicts, ns,
          predm, cumlens, cumnparts, thistset)
  return trial_tset, tsets, predm, nparts, cumnparts, cumlens


#####################################

# Miscellaneous utility functions


def findMinima(rdict, beta, rhomax):
  """
  Utility function to extract list of minima in rho for given beta, that
    lies below rhomax.

  Returns:
    - nmins: List of n at which rho is minimum
  """
  nvals = []
  rvals = []
  for b, n in rdict:
    if b == beta:
      nvals.append(n)
      rvals.append(rdict[b, n])
  args = np.argsort(nvals)
  nvals = np.array(nvals)[args]
  rvals = np.array(rvals)[args]
  nlen = len(nvals)
  nmins = []
  for i in range(nlen):
    if (rvals[i] < rhomax) and \
       (i == 0 or rvals[i] < rvals[i-1]) and \
       (i == nlen - 1 or rvals[i] < rvals[i+1]):
          nmins.append(nvals[i])
  return nmins


def calculateGoutGin(cmat, tset):
  """
  Utility function to extract metastability measures for a given hub set.

  Returns: (gammaout, gammain, rho)
    - gammaout: Maximum gamma between hubs
    - gammain:  Minimum gamma into nearest hubs
    - rho:      gammaout / gammain
  """
  nbins = len(cmat)
  tcomp = list(set(range(nbins)) - set(tset))
  gammaout = min([np.max(cmat[:, tset][tset]), 1])
  gammain = min([1, np.min(np.max(cmat[:, tset][tcomp], axis=1))])
  return gammaout, gammain, gammaout / gammain


def locateOriginalHub(mvec, thistset):
  """
  Find original hub in thistset corresponding to membership vector mvec.

  Returns:
    - t: index of original hub in L0 partition subarray if found, else None
  """
  singletonpartinds = np.sort(np.nonzero(mvec)[0])
  t = np.array(list(set(singletonpartinds).intersection(set(thistset))))
  if len(t) == 0:
    t = None
  else:
    t = t[0]
  return list(singletonpartinds).index(t)


def calculateGoutGinSingleton(c, cmat, mvec, t, trial_tset, thistset):
  """
  compute Gout, Gin, and rho corresponding to a singleton subpartitioning.
  Inputs:
    - c: L0 partition gamma subarray
    - cmat: Full chromosome gamma array
    - mvec: Full chromosome membership array for L0 partition
    - t: Index of original hub in L0 subarray
    - trial_tset: Combined full-chromosome hub set to be tested
    - thistset: Original full-chromosome hub set

  Returns: (gammaout, gammain, rho)
    - gammaout: Maximum gamma between hubs
    - gammain:  Minimum gamma into nearest hubs
    - rho:      gammaout / gammain
  """
  # Compute Gin
  gin = np.min((c + np.eye(len(c)))[:, t])
  # Compute Gout
  ## Identify element in trial_tset and thistset to omit, and remove
  singletonpartinds = np.sort(np.nonzero(mvec)[0])
  t2 = singletonpartinds[t]
  trial_tset2 = np.array([t for t in trial_tset if not t in singletonpartinds])
  gout = max(np.max(cmat[trial_tset2, t2]), np.max(cmat[t2, trial_tset2]))
  return gout, gin, gout / gin


def calculateGoutGinL1(cmats, tsets, cmat, thism, trial_tset, thistset):
  gdatas = []
  for ipart, (c, t) in enumerate(zip(cmats, tsets)):
    if len(t) > 1:
      gdatas.append(calculateGoutGin(c, t))
    else:
      mvec = thism[ipart]
      gdatas.append(calculateGoutGinSingleton(c, cmat, mvec, t,
              trial_tset, thistset))
  gdatas = np.array(gdatas).T
  gouts, gins, rhos = gdatas
  return gouts, gins, rhos


#####################################

# Metrics

def separationMetrics(arrays, tset, m):
  """
  Compute metrics of separation between partitions defined by membership
  matrix m:
  Inputs:
    - arrays = (fmat, mmat, cmat, mappingdata)
    - tset: target hub set
    - m: membership matrix (only makes sense for hard partitioning)

  Example:
  scores = separationMetrics((fmat, mmat, cmat, mappingdata), tset, m)
  print 'Metastability      :   gout=%.2e,   gin=%.2e,   rho   =%.2e' % tuple(scores[0])
  print 'Total flux         :   fout=%.2e,   fin=%.2e,   fratio=%.2e' % tuple(scores[1])
  print 'Escape probability :   pout=%.2e,   pin=%.2e,   pratio=%.2e' % tuple(scores[2])
  print 'Time scales        :   mout=%.2e,   min=%.2e,   mratio=%.2e' % tuple(scores[3])
  """
  fmat, mmat, cmat, mappingdata = arrays
  # Metastability gammas and ratio
  nbins = len(cmat)
  tcomp = list(set(range(nbins)) - set(tset))
  gammaout = min([np.max(cmat[:, tset][tset]), 1])
  gammain = min([1, np.min(np.max(cmat[:, tset][tcomp], axis=1))])
  rho = gammaout / gammain
  # Flux ratios
  boxmat = np.dot(m.T, m)
  compmat = 1.0 - boxmat
  fluxin = np.sum(boxmat * fmat)
  fluxout = np.sum(compmat * fmat)
  fratio = fluxout / fluxin
  # Escape probability
  mu = np.sum(fmat, axis=1) / np.sum(fmat)
  poutavg = np.dot(np.sum(compmat * fmat, axis=1) / np.sum(fmat, axis=1), mu)
  pinavg = 1 - poutavg
  pratio = poutavg / pinavg
  # Time scale separation
  minmout = np.min(mmat[compmat > 0.0])
  maxmin = np.min(mmat[boxmat > 0.0])
  mratio = minmout / maxmin
  return (gammaout, gammain, rho), (fluxout, fluxin, fratio), \
         (poutavg, pinavg, pratio), (minmout, maxmin, mratio)


def printTruncTupleFloats(l):
  """
  Parse nested tuples into strings, floats converted to 3sf.
  """
  return tuple(
    ["{0:.3}".format(x) if isinstance(x, float) else
        (x if not isinstance(x, tuple) else printTruncTupleFloats(x))
          for x in l])


def printScores(scores):
  """
  Pretty printing of separation metrics.
  """
  s =  printTruncTupleFloats(scores)
  (gammaout, gammain, rho), (fluxout, fluxin, fratio), \
  (poutavg, pinavg, pratio), (minmout, maxmin, mratio) = s
  return ['%10s   %10s %10s %10s' % ('', 'Out', 'In', 'Out/In'),
          '%10s | %10s %10s %10s' % ('Gamma', gammaout, gammain, rho),
          '%10s | %10s %10s %10s' % ('Flux', fluxout, fluxin, fratio),
          '%10s | %10s %10s %10s' % ('Prob', poutavg, pinavg, pratio),
          '%10s | %10s %10s %10s' % ('MFPT', minmout, maxmin, mratio)]


def getData_nextlevelScores(datadir, cname, minind):
  fname = os.path.join(datadir, 'hierarchyData',
          'scoreData-%s-%02i.p' % (cname, minind))
  if os.path.isfile(fname):
    scoreData = pickle.load(open(fname, 'rb'))
  else:
    scoreData = {}
  return scoreData


def dumpData_nextlevelScores(datadir, cname, minind, scoreData):
  fname = os.path.join(datadir, 'hierarchyData',
          'scoreData-%s-%02i.p' % (cname, minind))
  pickle.dump(scoreData , open(fname, 'wb'))


def getData_nextlevelHierarchycases(datadir, cname, minind):
  fname = os.path.join(datadir, 'hierarchyData',
          'hierarchyData-%s-%02i.p' % (cname, minind))
  if os.path.isfile(fname):
    hierarchyData = pickle.load(open(fname, 'rb'))
  else:
    hierarchyData = {}
  return hierarchyData


def dumpData_nextlevelHierarchycases(datadir, cname, minind, hierarchyData):
  fname = os.path.join(datadir, 'hierarchyData',
          'hierarchyData-%s-%02i.p' % (cname, minind))
  pickle.dump(hierarchyData, open(fname, 'wb'))






