
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import cPickle as pickle
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
import rmsd

from ChromaWalker import ChromaWalker
import plotutils as plu
import matnorms as mn
import embed3d as etd
import rec3Dutils as rtd
import coordcomp as cc



debug = True
refineglobalrho = False

##############################
# Interaction resampling

def _resamplefmat_poisson(fmat, ratio):
    return np.random.poisson(fmat * ratio)


def _resamplefmat_normal(fmat, ratio, round=True):
    vals = np.random.normal(fmat, np.sqrt(fmat))
    return np.round(vals) if round else vals


def _filtersampledfmat(fmat, sigmapx):
    '''
    Apply gaussian filter to fmat
    '''
    # Get filtered fmat
    masksm = gaussian_filter((fmat > 0.0) * 1.0, sigmapx, mode='constant')
    masksm[masksm == 0.0] = np.inf
    return gaussian_filter(fmat, sigmapx, mode='constant') / masksm


##############################
# Global hierarchy

def _wholeGenomeRho(minsets, nselection):
    '''
    Whole genome metastability index.
    minsets['chr1'] = {n1: rho_{chr1,n1}, n2: rho_{chr1,n2}, ...}
    nselection is a dict with chr as key and the chosen n as value.
    '''
    score = np.average([minsets[cname][n]
                        for cname, n in nselection.iteritems()])
    return score

def metroProb(oldSelection, newSelection, minN, minsets, scoreFunc, kT=1.0):
    """
    Transition probability for Metropolis algorithm.
    """
    oldSize =  np.sum([v for (c, v) in oldSelection.iteritems()])
    newSize =  np.sum([v for (c, v) in newSelection.iteritems()])
    # If old selection is smaller than minN, always move upwards
    if (oldSize < minN) and (newSize > oldSize):
        return 1.0
    # Check new selection size above minN
    if newSize < minN:
        return -1.0
    # Compute scores
    oldScore = np.log(scoreFunc(minsets, oldSelection))
    newScore = np.log(scoreFunc(minsets, newSelection))
    # If score decrease
    if oldScore > newScore:
        return 1.0
    # If score increase
    else:
        return np.exp(-(newScore - oldScore) / kT)

def metroStep(prob):
    """
    Determine / roll dice to see if step is taken.
    """
    return np.random.random() < prob

def _getGlobalLevels_1(globalMinsMain, nvals, rvals, interval=100):
    """
    Extract minimum in rho over regular intervals.
    e.g., if interval = 100, get minima in ranges [0, 100), [100, 200), ...
          up to the highest interval where data is available
    """
    # Find data range and intervals to inspect
    nmin = np.min(nvals)
    nmax = np.max(nvals)
    levelmin = int(np.floor(nmin / float(interval)))
    levelmax = int(np.ceil(nmax / float(interval)))
    nchoices = []
    ns = []
    for lo in range(levelmin, levelmax):
        testmin = lo * interval
        testmax = (lo + 1) * interval
        rhomin = np.max(rvals)
        for nv, rv in zip(nvals, rvals):
            if nv >= testmin and nv < testmax and rhomin > rv:
                noptimal = nv
                rhomin = rv
        nchoices.append(globalMinsMain[noptimal][1])
        ns.append(noptimal)
    return ns, nchoices


##############################
# Fluctuation + Normalization

def fluctuateFab_multGauss(fab, strength, minfac=0.01):
    """
    Introduce fluctuation to fab matrix.
    """
    factor = np.random.normal(1.0, strength, fab.shape)
    factor = (factor + factor.T) / 2.0
    factor[factor < minfac] = minfac
    return fab * factor

def _get_normfunc(norm):
    if norm == 'raw':
        return mn.norm_raw
    elif norm == 'psize':
        return mn.norm_psize
    elif norm == 'SCN':
        return mn.norm_SCN0
    elif norm == 'SCPNs':
        return mn.norm_SCPNs
    elif norm == 'SCPNn':
        return mn.norm_SCPNn
    elif norm == 'rownorm1':
        return mn.norm_rownorm1
    elif norm == 'rownormIter':
        return mn.norm_rownormIter
    elif norm == 'rowsum1':
        return mn.norm_rowsum1
    elif norm == 'rowsumIter':
        return mn.norm_rowsumIter
    else:
        print 'Invalid norm %s!' % norm
        return



##############################
# Recon

def randomSample3DBall(ctr, r, N):
    """
    Sample N points in a 3D ball of radius r around center ctr.
    """
    # Triple normal vector for direction
    c = np.random.normal(0.0, 1.0, size=(N, 3))
    r2 = np.sqrt(np.sum(c ** 2, axis=1))
    # Radius rescaling
    r3 = np.random.uniform(0, 1, size=(N)) ** (1.0 / 3.0) / r2 * r
    # Shift
    c2 = (c.T * r3).T
    return c2 + ctr

def dumpVTF_Cloud1(psizes, rads, coords, npts=3000):
    """
    Sample single structure as a cloud of npts points.
    """
    if rads is None:
        rads = np.ones(len(psizes))
    # Write atoms
    natoms = map(int, np.ceil(psizes * float(npts) / np.sum(psizes)))
    iatom = 0
    cloudcoords = []
    partitionindex = []
    # Write atoms, create cloud
    for i, (natom, rad, ctr) in enumerate(zip(natoms, rads, coords)):
        cloudcoords.extend(list(randomSample3DBall(ctr, rad, natom)))
        partitionindex.extend([i] * natom)
    return np.array(cloudcoords), np.array(partitionindex)

## Aligning across partitioning levels using set of identical partitions

def _findIdenticalParts(plims, chains, cnamelist):
    pl1, pl2 = plims
    ch1, ch2 = chains
    ch1 = np.array(ch1)
    ch2 = np.array(ch2)
    pl1 = np.array(pl1)
    pl2 = np.array(pl2)
    shift1 = 0
    shift2 = 0
    matches = []
    for c in cnamelist:
        cmask1 = (ch1 == c[3:])
        cmask2 = (ch2 == c[3:])
        thispl1 = pl1[cmask1]
        thispl2 = pl2[cmask2]
        for i, x in enumerate(thispl1):
            for j, y in enumerate(thispl2):
                if np.allclose(x, y):
                    matches.append([i + shift1, j + shift2])
        shift1 += len(thispl1)
        shift2 += len(thispl2)
    return np.array(matches)

def findIdenticalParts(partitions2, partitionsref, cnamelist):
    pl1 = []
    ch1 = []
    pl2 = []
    ch2 = []
    for ch, pl in partitionsref:
        pl2.append(pl)
        ch2.append(ch[3:])
    for ch, pl in partitions2:
        pl1.append(pl)
        ch1.append(ch[3:])
    return _findIdenticalParts((pl1, pl2), (ch1, ch2), cnamelist)

def _AlignCoords_IdenticalParts(coordsRef, coords, identicalInds,
        weights=None, getrmsd=True):
    """
    Align coords to coordsRef by considering only identical partitions,
    with indices in the list identicalInds:
      for i, j in identicalInds:
        coords[i] corresponds to coordsRef[j]
    Vector weights correspond to partitions in coords.
    """
    c1 = coords[identicalInds[:, 0]]
    c2 = coordsRef[identicalInds[:, 1]]
    ############################
    if weights is None:
        weights = 1.0
        w1 = weights
        w2 = weights
    else:
        w1 = np.sqrt(weights[0][identicalInds[:, 0]])
        w2 = np.sqrt(weights[1][identicalInds[:, 1]])
    data = [0 for i in range(8)]
    datar = [0 for i in range(8)]
    for i in range(2):
        x = c1[:, 0] * (-1) ** i
        x1 = coords[:, 0] * (-1) ** i
        for j in range(2):
            y = c1[:, 1] * (-1) ** j
            y1 = coords[:, 1] * (-1) ** j
            for k in range(2):
                z = c1[:, 2] * (-1) ** k
                z1 = coords[:, 2] * (-1) ** k
                ind = ((i * 2) + j) * 2 + k
                data[ind] = np.array([x1, y1, z1]).T
                datar[ind] = np.array([x * w1,
                                       y * w1,
                                       z * w1]).T
    coordsr2 = c2 * np.tile(w2, (3, 1)).T
    rmsdcheck = [rmsd.kabsch_rmsd(c, coordsr2) for c in datar]
    igood = np.argmin(rmsdcheck)
    #print 'igood', igood
    rotmat = rmsd.kabsch(datar[igood], coordsr2)
    #print rotmat
    #print det(rotmat)
    rotcoords = np.dot(data[igood], rotmat)
    if getrmsd:
        return rotcoords, cc.RMSD(rotcoords[identicalInds[:, 0]], c2, weights=w1)
    else:
        return rotcoords

def AlignCoords_IdenticalParts(coordsref, coords2,
                               partitionsref, partitions2,
                               psizesref, psizes2,
                               cnamelist,
                               getrmsd=True):
    """
    Align structure at different partitioning levels.
    Using coordsref as reference, output aligned coords2.
    Alignment performed by matching identical partitions only.
    """
    identicalInds = findIdenticalParts(partitions2, partitionsref, cnamelist)
    if len(identicalInds) == 0:
        print 'AlignCoords_IdenticalParts: No identical partitions found! Not aligning...'
        return (coords2, np.zeros_like(coords2)) if getrmsd else (coords2)
    return _AlignCoords_IdenticalParts(coordsref, coords2, identicalInds,
                               weights=[psizes2, psizesref], getrmsd=getrmsd)


## Dumping VTF files

def startVTF_psizes(fname, psizes, chains, bonds=False, sizepow=0.6):
    """
    Create VTF file to store structure.
    Returns file pointer.
    """
    fp = open(fname, 'w')
    #psizes = np.sum(membership, axis=1)
    nparts = len(psizes)
    # Write atoms
    for ipart in range(nparts):
        fp.write('a %i radius %.2lf s %s\n' %
                (ipart, psizes[ipart] ** sizepow, chains[ipart]))
    # Write bonds
    if bonds:
      for ipart in range(nparts - 1):
          if chains[ipart] == chains[ipart + 1]:
              fp.write('b %i:%i\n' % (ipart, ipart + 1))
    return fp

def appendVTF(fp, coords):
    """
    Add new timestep to VTF file
    """
    fp.write('timestep ordered\n')
    for c in coords:
        fp.write('%.2lf %.2lf %.2lf\n' % tuple(c))



##############################
# Reporting

## Rescaled RMSD between two clouds

## Distance correlation: single-chr and chr-pair

def getDists_instance(cloudcoords, mask, maxptsamples=10000):
    """
    Get list of inter-point distances at the same instances.
    """
    ntrials = len(cloudcoords)
    ds = [None for itrial in range(ntrials)]
    for itrial in range(ntrials):
        selinds = np.nonzero(mask)[0]
        if len(selinds) < maxptsamples:
            inds = selinds
        else:
            inds = np.sort(np.random.choice(selinds, maxptsamples, replace=False))
        cc = cloudcoords[itrial, inds]
        ds[itrial] = cdist(cc, cc)[np.triu_indices(len(cc), k=1)]
    ds = np.array(ds).flatten()
    return ds


def getDists_ensemble(cloudcoords, mask, maxptsamples=10000):
    """
    Get list of inter-point distances across all instances.
    """
    ntrials = len(cloudcoords)
    selinds = np.nonzero(list(mask) * ntrials)[0]
    if len(selinds) < maxptsamples:
        inds = selinds
    else:
        inds = np.sort(np.random.choice(selinds, maxptsamples, replace=False))
    ccf = cloudcoords.flatten()
    ccf.shape = len(ccf) / 3, 3
    cc = ccf[inds]
    ds= cdist(cc, cc)[np.triu_indices(len(cc), k=1)]
    return ds

## Chromosome proximity index

def getCrosschrOverlapratios(cloudcoords, mask1, mask2, distcutofflist, maxptsamples=10000):
    """
    Get list of inter-point distances across all instances.
    """
    ntrials = len(cloudcoords)
    selinds1 = np.nonzero(list(mask1) * ntrials)[0]
    selinds2 = np.nonzero(list(mask2) * ntrials)[0]
    n1 = len(selinds1)
    n2 = len(selinds2)
    m1 = int(float(maxptsamples) * float(n1) / (n1 + n2))
    m2 = int(float(maxptsamples) * float(n2) / (n1 + n2))
    inds1 = selinds1 if (n1 < m1) else np.sort(np.random.choice(selinds1, m1, replace=False))
    inds2 = selinds2 if (n2 < m2) else np.sort(np.random.choice(selinds2, m2, replace=False))
    ccf = cloudcoords.flatten()
    ccf.shape = len(ccf) / 3, 3
    cc1 = ccf[inds1]
    cc2 = ccf[inds2]
    ds12 = cdist(cc1, cc2).flatten()
    npairclose = []
    #print 'mindist =', np.min(ds12)
    for d in distcutofflist:
        npairclose.append(np.sum(ds12 <= d))
    fracclose = np.array(npairclose) / float(len(ds12))
    return fracclose


##############################
# ChromaSEP

class ChromaSEP:

    def __init__(self, pars, cw):
        print 'Initializing ChromaSEP instance...'
        # Basic pars and directories
        ## Data
        self.res = pars['res']
        self.cnamelist = pars['cnamelist']
        self.rhomax = pars['rhomax']
        self.reconlabel = pars['reconlabel']
        self.sampleninterval = pars['sampleninterval']
        ## Resampling interactions
        self.resample = pars.get('resample', False)
        self.resample_ratio = pars.get('resample_ratio', 1.0)
        self.gfilterpx = 0.0 if (pars['fijnorm'][:7] != 'gfilter') \
                             else (float(pars['fijnorm'][8:]) / self.res)
        ## Normalization
        self.intranorm = pars.get('intranorm', 'Pow')
        self.norm = pars['norm']
        self.normfunc = _get_normfunc(self.norm)
        ## Reconstruction
        self.embedargs = pars['embedargs']
        self.nalphasamples = pars['nalphasamples']
        self.alphavals = pars['alphavals']
        self.nSEPpoints = pars['nSEPpoints']
        ## Directories
        self.datadir = os.path.join(self.reconlabel, pars['datadir'])
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)
        self.plotdir = os.path.join(self.reconlabel, pars['plotdir'])
        if not os.path.isdir(self.plotdir):
            os.makedirs(self.plotdir)
        # ChromaWalker instance
        self.cw = cw
        ##################################
        # Find base beta
        print 'Finding base beta...'
        if 'basebeta' in  pars:
            self.basebeta = pars['basebeta']
        elif 'betaselection' in pars:
            self.basebeta = np.min([pars['betaselection'][cname]
                for cname in self.cnamelist])
        else:
            self.basebeta = np.min([cw._get_bestBeta(cname)
                for cname in self.cnamelist])
        print 'Base beta = %i .' % int(self.basebeta)
        print

    def _getAllGoodLevels(self, debug=False):
        '''
        For each chromosome, extract rho, n for all good levels n.
        '''
        print '> Get all good single-chr levels...'
        self.goodNLevels_chr = {}
        # For each chromosome...
        for cname in self.cnamelist:
            ## Get rdict
            rdict, tdict = self.cw.DFR.get_datadicts(cname)
            ntlist = []
            rlist = []
            n = 2
            while True:
                data = self.cw.DFR.readout_datadicts(rdict, tdict, self.basebeta, n)
                if data is not None:
                    ntlist.append(n)
                    rlist.append(data[0])
                    n += 1
                else:
                    break
            if debug:
                print cname, 'max n:', np.max(ntlist)
            ## Save all good levels and corresponding rho
            self.goodNLevels_chr[cname] = {}
            for i, n in enumerate(ntlist):
                if i == 0:
                    continue
                elif i == len(ntlist) - 1 and ((rlist[i] > self.rhomax) or
                                               (rlist[i] > rlist[i-1])):
                    continue
                elif ((rlist[i] < self.rhomax) and
                      (rlist[i] < rlist[i-1]) and
                      (rlist[i] < rlist[i+1])):
                    self.goodNLevels_chr[cname][n] = rlist[i]
            if debug:
                print cname, 'good levels:', self.goodNLevels_chr[cname]

    def _getGlobalRhosMain(self, niters=10, debug=False, refine=False):
        '''
        Use niters rounds of MC sampling at each kT level to
            optimize global rho profile.
        '''
        print 'Getting global rho trend...'
        fname = os.path.join(self.datadir, 'globalRhosMain.p')
        if os.path.isfile(fname):
            self.globalRhosMain = pickle.load(open(fname, 'r'))
            if not refine:
                return
        else:
            self.globalRhosMain = {}
        # Initialize parameters
        minN0 = np.sum([np.min(self.goodNLevels_chr[cname].keys())
                        for cname in self.cnamelist])              # Sum of all minimum n's
        maxN0 = np.sum([np.max(self.goodNLevels_chr[cname].keys())
                        for cname in self.cnamelist]) - 10         # Sum of all maximum n's
        kTvals = [1.0, 0.5, 0.2, 0.1, 0.02, 0.01, 0.001]
        kTvals = np.sort(kTvals * niters)[::-1]
        nstepstabilize = 200
        nsteptrack = 1000
        # Test with different values of kT
        for kT in kTvals:
            globalMins = {}
            nadds = 0
            nimprovements = 0
            minN = minN0
            while minN < maxN0:
                # Initialize
                thisSelection = {c: np.random.choice(self.goodNLevels_chr[c].keys()) for c in cnamelist}
                # Stabilize
                for i in range(nstepstabilize):
                    newSelection = copy.deepcopy(thisSelection)
                    changeC = np.random.choice(cnamelist)
                    newSelection[changeC] = np.random.choice(self.goodNLevels_chr[changeC].keys())
                    if metroStep(metroProb(thisSelection, newSelection, minN, self.goodNLevels_chr, _wholeGenomeRho, kT=kT)):
                        thisSelection = copy.deepcopy(newSelection)
                # Track
                tracker = []
                for i in range(nsteptrack):
                    newSelection = copy.deepcopy(thisSelection)
                    changeC = np.random.choice(cnamelist)
                    newSelection[changeC] = np.random.choice(self.goodNLevels_chr[changeC].keys())
                    if metroStep(metroProb(thisSelection, newSelection, minN, self.goodNLevels_chr, _wholeGenomeRho, kT=kT)):
                        thisSelection = copy.deepcopy(newSelection)
                    tracker.append([np.sum([v for (c, v) in thisSelection.iteritems()]),
                                    _wholeGenomeRho(self.goodNLevels_chr, thisSelection),
                                    copy.deepcopy(thisSelection)])
                # Analyze
                ## Minimum N
                thisNs = np.array([v[0] for v in tracker])
                if np.max(thisNs) < minN:
                    break
                thisScores = np.array([v[1] for v in tracker])
                thisminN = np.min(thisNs[thisNs >= minN])
                minargs = np.nonzero(thisNs == thisminN)[0]
                if len(minargs) > 1:
                    # Find the one that minimizes score (v[1])
                    thisarg = minargs[np.argmin(thisScores[minargs])]
                else:
                    thisarg = minargs[0]
                globalMins[thisminN] = copy.deepcopy((tracker[thisarg][1], tracker[thisarg][2]))
                # Update globalMinsMain
                if thisminN not in self.globalRhosMain:
            #         print 'Add N =', thisminN
                    self.globalRhosMain[thisminN] = copy.deepcopy((tracker[thisarg][1], tracker[thisarg][2]))
                    nadds += 1
                elif globalMins[thisminN][0] < self.globalRhosMain[thisminN][0]:
            #         print 'Improved N =', thisminN
                    self.globalRhosMain[thisminN] = copy.deepcopy((tracker[thisarg][1], tracker[thisarg][2]))
                    nimprovements += 1
                allNs = np.sort(self.globalRhosMain.keys())
                if len(allNs[allNs > minN]) == 0:
                    nextminN = thisminN + 1
                else:
                    nextminN = min(thisminN + 1, np.min(allNs[allNs > minN]))
                if int(minN / 100) != int(nextminN / 100):
                    print 'Next minN:', nextminN
                minN = nextminN
        #         print 'Next minN:', minN
            print 'This round kT:', kT, 'add %3i improve %3i' % (nadds, nimprovements)
            fname = os.path.join(self.datadir, 'globalRhosMain.p')
            pickle.dump(self.globalRhosMain, open(fname, 'w'))
            print 'Updated globalRhosMain.p...'

    def _getPartitionData(self, chrnchoices):
        '''
        Get genome-wide partitioning data for the given choices of n.
        '''
        pass
        # For each chromosome...
        partition_chrinds = []
        for cname in self.cnamelist:
            ## Get partition info
            memb, lims, inds = self.cw.TOpt.get_partitions(
                      cname, self.basebeta, chrnchoices[cname])
            ## Append data
            for lim, ind in zip(lims, inds):
                if inds >= 0:
                    partition_chrinds.append([cname, lim])
        return partition_chrinds

    def getGlobalHierarchy(self, debug=False):
        print 'Getting global structural hierarchy...'
        # Get single-chr rho data
        self._getAllGoodLevels(debug=debug)
        # Get globalRhosMain
        self._getGlobalRhosMain(debug=debug, refine=refineglobalrho)
        fname = os.path.join(self.datadir, 'globalHierarchyData.p')
        if os.path.isfile(fname) and not refineglobalrho:
            data = pickle.load(open(fname, 'r'))
            self.globalNs = data['globalNs']
            self.nchoices = data['nchoices']
            self.partitiondata = data['partitiondata']
        else:
            # Get global rho trend
            nlist = list(np.sort(self.globalRhosMain.keys()))
            rlist = [self.globalRhosMain[n][0] for n in nlist]
            # Get selected globalNs
            self.globalNs, self.nchoices = _getGlobalLevels_1(
                      self.globalRhosMain, nlist, rlist, interval=self.sampleninterval)
            print 'GlobalNs:', self.globalNs
            # Get selected globalNs, chrNchoices, and corresponding partitioning info
            self.partitiondata = {n: self._getPartitionData(nc) for n, nc in zip(self.globalNs, self.nchoices)}
            # Save hierarchy data to file
            data = {
              'globalNs': self.globalNs,
              'nchoices': self.nchoices,
              'partitiondata': self.partitiondata
              }
            fname = os.path.join(self.datadir, 'globalHierarchyData.p')
            pickle.dump(data, open(fname, 'w'))
        self.psizes = {n: np.array([(en - st) for c, (st, en) in pd])
                          for (n, pd) in self.partitiondata.iteritems()}
        if debug:
            print 'First level psizes:', self.psizes[0]

    def getBaseFab(self):
        print 'Getting unperturbed effective interaction matrix...'
        fname = os.path.join(self.datadir, 'basefabs.p')
        if os.path.isfile(fname):
            self.basefabs = pickle.load(open(fname, 'r'))
            return
        self.basefabs = {}
        # Binning info
        print '> Getting bin info...'
        indsss = {}
        membss = {}
        for ilevel, globalN in enumerate(self.globalNs):
            nparts = len(self.partitiondata[globalN])
            self.basefabs[globalN] = np.zeros((nparts, nparts))
            membs = {}
            indss = {}
            for cname in self.cnamelist:
                inds = []
                memb = []
                _, nbins = self.cw.DFR.get_mappingdata(cname, 1.0)
                for i, (cn, (st, en)) in enumerate(self.partitiondata[globalN]):
                    if cn == cname:
                        inds.append(i)
                        memb.append(np.zeros(nbins))
                        memb[-1][st:en] = 1.0
                inds = np.array(inds)
                memb = np.array(memb)
                indss[cname] = inds.copy()
                membs[cname] = memb.copy()
            indsss[globalN] = indss
            membss[globalN] = membs
        print '> Getting intra-chr interaction data...'
        # Intra-chr
        for cname in self.cnamelist:
            # Get fmat,mappingdata
            fmat = self.cw.DFR.get_fmat(cname, 1.0)
            mappingdata = self.cw.DFR.get_mappingdata(cname, 1.0)
            # Resample
            if self.resample:
                fmat = _resamplefmat_poisson(fmat, self.resample_ratio)
            # Intra-norm
            allDist, allFij, dist, avgVsDist, nAtDist = mn.preprocess_fmat(
                      fmat, mappingdata, plot=False)
            if self.intranorm == 'Pow':
                fmatp = mn.fmatRescale_powlaw(fmat, mappingdata, allDist, allFij, dist, nb=100, plot=False)
            elif self.intranorm == 'Ratio':
                fmatp = mn.fmatRescale_ratio(fmat, mappingdata, dist, avgVsDist)
            elif self.intranorm != 'Raw':
                print 'Unrecognized intra norm %s! Ignoring intra-norm...' % self.intranorm
                fmatp = fmat
            fmatpad = plu._build_fullarray(fmatp, mappingdata, 0.0)
            for globalN in self.globalNs:
                # Get partitioning data
                memb = membss[globalN][cname]
                inds = indsss[globalN][cname]
                # Get submat
                submat = np.dot(np.dot(memb, fmatpad), memb.T)
                submat -= np.diag(np.diag(submat))
                for i, ind in enumerate(inds):
                    self.basefabs[globalN][ind, inds] = submat[i]
        print '> Getting inter-chr interaction data...'
        # Inter-chr
        for i1, cname1 in enumerate(self.cnamelist):
            for cname2 in self.cnamelist[i1+1:]:
                # Get fmat, mappingdata
                fmat, md1, md2 = self.cw.DFR.get_fmatMapdata_inter(cname1, cname2)
                # Resample
                if self.resample:
                    fmat = _resamplefmat_poisson(fmat, self.resample_ratio)
                fmatpad = plu._build_fullarray_inter(fmat, md1, md2, 0.0)
                for globalN in self.globalNs:
                    # Get partitioning data
                    memb1 = membss[globalN][cname1]
                    inds1 = indsss[globalN][cname1]
                    memb2 = membss[globalN][cname2]
                    inds2 = indsss[globalN][cname2]
                    # Get submat
                    submat = np.dot(np.dot(memb1, fmatpad), memb2.T)
                    for i, ind in enumerate(inds1):
                        self.basefabs[globalN][ind, inds2] = submat[i]
                    for i, ind in enumerate(inds2):
                        self.basefabs[globalN][ind, inds1] = submat[:, i]
        print '> Calculated base Fab data.'
        # Remove zero-interaction partitions
        for globalN in self.globalNs:
            # Find good inds
            goodinds = np.nonzero(np.sum(self.basefabs[globalN], axis=0) > 0)[0]
            # Keep only good elements in self.partitiondata
            self.partitiondata[globalN] = [
                pd for i, pd in enumerate(self.partitiondata[globalN])
                    if i in goodinds]
            # Keep only good elements in self.psizes
            self.psizes[globalN] = [
                ps for i, ps in enumerate(self.psizes[globalN])
                    if i in goodinds]
            # Keep only good elements in self.basefab
            self.basefabs[globalN] = self.basefabs[globalN][goodinds][:, goodinds]
        # Update globalHierarchyData.p
        data = {
          'globalNs': self.globalNs,
          'nchoices': self.nchoices,
          'partitiondata': self.partitiondata
          }
        fname = os.path.join(self.datadir, 'globalHierarchyData.p')
        pickle.dump(data, open(fname, 'w'))
        print '> Updated globalHierarchyData.p'
        # Save Fab matrix to file
        fname = os.path.join(self.datadir, 'basefabs.p')
        pickle.dump(self.basefabs, open(fname, 'w'))
        print '> Saved base Fab data.'

    def getBaseRecon(self):
        print 'Getting unperturbed reconstruction at partition level...'
        fname = os.path.join(self.datadir, 'baserecons.p')
        if os.path.isfile(fname):
            self.baserecons = pickle.load(open(fname, 'r'))
            return
        self.baserecons = {}
        for ilevel, globalN in enumerate(self.globalNs):
            print '> Generating base recon for globalN =', globalN, '...'
            # Get dab
            fab = self.basefabs[globalN]
            psizes = self.psizes[globalN]
            fabp = self.normfunc(fab, psizes=psizes)
            dab = mn.fab2dab(fabp, **embedargs)
            # Get reconstruction, centered and rescaled
            coords, evals = etd.dist2Recon(dab, **self.embedargs)
            coords = cc.centerRescaleCoords(coords, self.embedargs['sizeScale'])
            # Align to lower levels
            if ilevel > 0:
                coords = AlignCoords_IdenticalParts(
                          self.baserecons[self.globalNs[ilevel-1]], coords,
                          self.partitiondata[self.globalNs[ilevel-1]], self.partitiondata[globalN],
                          self.psizes[self.globalNs[ilevel-1]], psizes,
                          self.cnamelist, getrmsd=False)
            self.baserecons[globalN] = coords.copy()
        fname = os.path.join(self.datadir, 'baserecons.p')
        pickle.dump(self.baserecons, open(fname, 'w'))
        print 'Saved base recons data.'

    def getBaseReconVTF(self):
        print 'Producing VTF files for base recons...'
        # Create dump directory
        filedir = os.path.join(self.datadir, 'baserecons')
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        # For each level, read psizes, chains and coords
        for ilevel, globalN in enumerate(self.globalNs):
            psizes = self.psizes[globalN]
            chains = [c[3:] for c, _ in self.partitiondata[globalN]]
            coords = self.baserecons[globalN]
            fname = os.path.join(filedir, 'N%03i.vtf' % globalN)
            with startVTF_psizes(fname, psizes, chains) as fp:
                appendVTF(fp, coords)

    def getSEPRecon(self):
        print 'Getting SEP reconstructions...'
        # For each level, generate alpharecon aligned to baserecon and SEPcloud coords
        alphadir = os.path.join(self.datadir, 'alpharecons')
        sepdir = os.path.join(self.datadir, 'SEPrecons')
        if not os.path.isdir(alphadir):
            os.makedirs(alphadir)
        if not os.path.isdir(sepdir):
            os.makedirs(sepdir)
        self.alpharecons = {}
        print '> Getting alpha (noisy Fab, partition-level) reconstructions...'
        for globalN in self.globalNs:
            # alpharecons: Perturbed Fab, without sampling in partitions
            fname = os.path.join(alphadir, 'N%03i.p' % globalN)
            if os.path.isfile(fname):
                self.alpharecons[globalN] = pickle.load(open(fname, 'r'))
                continue
            # Generate perturbed Fab and embed partitions
            self.alpharecons[globalN] = {}
            fab = self.basefabs[globalN]
            psizes = self.psizes[globalN]
            basecoords = self.baserecons[globalN]
            for alpha in self.alphavals:
                print '>>> Running alpharecons for globalN=%i, alpha=%s' % (globalN, str(alpha))
                self.alpharecons[globalN][alpha] = np.zeros(
                          (self.nalphasamples, len(psizes), 3))
                for ipertF in range(self.nalphasamples):
                    # Perturb using alpha
                    fab2 = fluctuateFab_multGauss(fab, alpha, minfac=0.01)
                    # Embed partitions
                    fabp = self.normfunc(fab2, psizes=psizes)
                    dab = mn.fab2dab(fabp, **self.embedargs)
                    coords, evals = etd.dist2Recon(dab, **self.embedargs)
                    # Get reconstruction, centered and rescaled, and aligned to baserecon
                    coords, evals = etd.dist2Recon(dab, **self.embedargs)
                    coords = cc.centerRescaleCoords(coords, self.embedargs['sizeScale'])
                    coords = rtd.alignCoords(coords, basecoords, weights=psizes)
                    self.alpharecons[globalN][alpha][ipertF] = coords
            fname = os.path.join(alphadir, 'N%03i.p' % globalN)
            pickle.dump(self.alpharecons[globalN], open(fname, 'w'))
            print '>> Saved alpha recon data at N = %i...' % globalN
        self.SEPrecons = {}
        print '> Getting full SEP reconstructions...'
        for globalN in self.globalNs:
            # SEPrecons: Full SEP clouds
            fname = os.path.join(sepdir, 'N%03i.p' % globalN)
            if os.path.isfile(fname):
                self.SEPrecons[globalN] = pickle.load(open(fname, 'r'))
                continue
            psizes = self.psizes[globalN]
            rads = (psizes * self.embedargs['radius_factor']) ** self.embedargs['radius_exp']
            # Generate SEP clouds
            self.SEPrecons[globalN] = {}
            for alpha in self.alphavals:
                print '>>> Running SEPrecons for globalN = %i, alpha = %s' % (globalN, str(alpha))
                self.SEPrecons[globalN][alpha] = []
                for ipertF in range(self.nalphasamples):
                    coords1 = self.alpharecons[globalN][alpha][ipertF]
                    cloudcoords, partitionindex = dumpVTF_Cloud1(
                          psizes, rads, coords1, npts=self.nSEPpoints)
                    self.SEPrecons[globalN][alpha].append(cloudcoords.copy())
                self.SEPrecons[globalN][alpha] = np.array(self.SEPrecons[globalN][alpha])
            self.SEPrecons[globalN]['partitionindex'] = partitionindex
            fname = os.path.join(sepdir, 'N%03i.p' % globalN)
            pickle.dump(self.SEPrecons[globalN], open(fname, 'w'))
            print '>> Saved SEP recon data at N = %i...' % globalN

    def getSEPReconVTF(self):
        print 'Producing VTF files for SEP recons...'
        # Create dump directory
        sepdir = os.path.join(self.datadir, 'SEPrecons')
        self.SEPrecons = {}
        for globalN in self.globalNs:
            # SEPrecons: Full SEP clouds
            print '> Getting SEP recon at globalN = %i' % globalN
            fname = os.path.join(sepdir, 'N%03i.p' % globalN)
            self.SEPrecons[globalN] = pickle.load(open(fname, 'r'))
            partitionindex = self.SEPrecons[globalN]['partitionindex']
            chains = [self.partitiondata[globalN][i][0][3:] for i in partitionindex]
            ##################
            for alpha in self.alphavals:
                print '>> Getting SEP recon at globalN = %i, alpha = %s' % (globalN, str(alpha))
                coords = self.SEPrecons[globalN][alpha]
                fname = os.path.join(sepdir, 'N%03i-alpha%s.vtf' % (globalN, str(alpha)))
                with startVTF_psizes(fname, np.ones(len(coords[0])), chains) as fp:
                    for c in coords:
                        appendVTF(fp, c)

    def getReportsHierarchy(self):
        print 'Generating reports for global hierarchy...'
        # Get global rho trend
        nlist = list(np.sort(self.globalRhosMain.keys()))
        rlist = [self.globalRhosMain[n][0] for n in nlist]
        # Get selected globalNs
        globalNs, nchoices = _getGlobalLevels_1(
                  self.globalRhosMain, nlist, rlist, interval=50)
        # Plot
        f, x = plt.subplots(figsize=(16, 8))
        _ = x.plot(nlist, rlist)
        rs = [rlist[nlist.index(n)] for n in globalNs]
        _ = x.scatter(globalNs, rs, color='r', s=5)
        _ = x.set_title('Global metastability index $\\rho$', fontsize=16)
        _ = x.set_xlabel('$N$')
        _ = x.set_ylabel('$\\rho$')
        fname = os.path.join(self.plotdir, 'GlobalMetastabilityIndex.pdf')
        f.savefig(fname)
        plt.close(f)

    def getReportsSEP(self):
        print 'Generating reports for SEP reconstructions...'
        # Rescaled RMSD comparison between levels
        print '> Inter-level rescaled RMSDs...'
        ## Get intra-level RMSDs
        fname = os.path.join(self.datadir, 'rmsd_intralevel.p')
        if os.path.isfile(fname):
            self.rmsd_intralevel = pickle.load(open(fname, 'r'))
        else:
            self.rmsd_intralevel = {}
            for globalN in self.globalNs:
                for alpha in self.alphavals:
                    # Load cloudcoords
                    cloudcoords = self.SEPrecons[globalN][alpha]
                    # Estimate average RMSD between SEP recons by comparing consecutive recons
                    dists = [cc.RMSD(cloudcoords[i], cloudcoords[i+1])
                             for i in range(self.nalphasamples - 1)]
                    # Store intra-level RMSD
                    self.rmsd_intralevel[globalN, alpha] = np.average(dists)
            fname = os.path.join(self.datadir, 'rmsd_intralevel.p')
            pickle.dump(self.rmsd_intralevel, open(fname, 'w'))
        ## Get inter-level RMSDs
        fname = os.path.join(self.datadir, 'rmsd_interlevel.p')
        if os.path.isfile(fname):
            self.rmsd_interlevel = pickle.load(open(fname, 'r'))
        else:
            self.rmsd_interlevel = {}
            for i, globalN1 in enumerate(self.globalNs):
                for globalN2 in self.globalNs[i+1:]:
                    for alpha in self.alphavals:
                        cloudcoords1 = self.SEPrecons[globalN1][alpha]
                        cloudcoords2 = self.SEPrecons[globalN2][alpha]
                        n = min(len(cloudcoords1[0]), len(cloudcoords2[0]))
                        self.rmsd_interlevel[globalN1, globalN2, alpha] = \
                            np.average([cc.RMSD(cloudcoords1[i, :n], cloudcoords2[i, :n])
                                        for i2 in range(self.nalphasamples)])
                        self.rmsd_interlevel[globalN2, globalN1, alpha] = \
                            self.rmsd_interlevel[globalN1, globalN2, alpha]
            fname = os.path.join(self.datadir, 'rmsd_interlevel.p')
            pickle.dump(self.rmsd_interlevel, open(fname, 'w'))
        ## Generate rescaled RMSD plots
        fname = os.path.join(self.plotdir, 'RescaledRMSDs.pdf')
        with PdfPages(fname) as pdf:
            for alpha in self.alphavals:
                f, x = plt.subplots(figsize=(10, 8))
                dmat = [[1.0 if globalN1 == globalN2 else
                         (self.rmsd_interlevel[globalN1, globalN2, alpha] /
                          np.sqrt(self.rmsd_intralevel[globalN1, alpha] *
                                  self.rmsd_intralevel[globalN2, alpha]))
                         for globalN1 in self.globalNs]
                        for globalN2 in self.globalNs]
                img = x.imshow(dmat, vmin=1)
                x.set_xlabel('$N_1$')
                x.set_ylabel('$N_2$')
                x.set_title('Rescaled RMSD at $\\alpha=%s$' % str(alpha))
                x.set_ylim(x.get_ylim()[::-1])
                x.set_xticks(range(len(self.globalNs)))
                x.set_yticks(range(len(self.globalNs)))
                x.set_xticklabels(self.globalNs, rotation='vertical')
                x.set_yticklabels(self.globalNs)
                cbar = plt.colorbar(img, ax=x)
                pdf.savefig(f)
                plt.close(f)
        # Single-chr distance correlation plots at the finest level, highest alpha
        print '> Single-chr distance correlations at highest N and alpha...'
        globalN = self.globalNs[-1]
        alpha = self.alphavals[-1]
        partitionindex = self.SEPrecons[globalN]['partitionindex']
        chains = np.array([self.partitiondata[globalN][i][0]
                           for i in partitionindex])
        cloudcoords = self.SEPrecons[globalN][alpha]
        maxdist = np.max(np.abs(cloudcoords))
        ndistsamples = 200
        distbins = np.linspace(0.0, maxdist, ndistsamples + 1)
        distbinctrs = (distbins[:-1] + distbins[1:]) / 2.0
        fname = os.path.join(self.plotdir, 'DistCorrelations-singlechr.pdf')
        with PdfPages(fname) as pdf:
            for cname in self.cnamelist:
                mask = (chains == cname)
                ds = getDists_instance(cloudcoords, mask, maxptsamples=10000)
                hist, _ = np.histogram(ds, bins=distbins, density=True)
                f, x = plt.subplots(figsize=(10, 8))
                x.plot(distbinctrs, hist)
                x.set_xlabel('Length [a.u.]')
                x.set_ylabel('Probability')
                x.set_title(cname)
                pdf.savefig(f)
                plt.close(f)
        # Chromosome proximity index at highest N, highest alpha
        print '> Chromosomal proximity index at highest N and alpha...'
        ## Get proximity index array
        globalN = self.globalNs[-1]
        alpha = self.alphavals[-1]
        cloudcoords = self.SEPrecons[globalN][alpha]
        distcutofflist = [3.0]
        fname = os.path.join(self.datadir, 'ChromosomalProximityIndex.p')
        if os.path.isfile(fname):
            self.chrproximityindex = pickle.load(open(fname, 'r'))
        else:
            self.chrproximityindex = np.zeros((len(self.cnamelist), len(self.cnamelist)))
            for i, cname1 in enumerate(self.cnamelist):
                mask1 = (chains == cname1)
                for j, cname2 in enumerate(self.cnamelist):
                    if i < j:
                        continue
                    mask2 = (chains == cname2)
                    self.chrproximityindex[i, j] = getCrosschrOverlapratios(
                            cloudcoords, mask1, mask2, distcutofflist)[0]
                    self.chrproximityindex[j, i] = self.chrproximityindex[i, j]
            fname = os.path.join(self.datadir, 'ChromosomalProximityIndex.p')
            pickle.dump(self.chrproximityindex, open(fname, 'w'))
        ## Cluster
        d = self.chrproximityindex.copy()
        v = np.sqrt(np.sum(d**2, axis=1))
        d2 = d / np.sqrt(np.outer(v, v) + sys.float_info.epsilon)
        fig = plt.figure(figsize=(10, 6))
        axdendro = fig.add_axes([0.09,0.1,0.17,0.8])
        Y = sch.linkage(d2, method='centroid')
        Z = sch.dendrogram(Y, orientation='left')
        axdendro.set_xticks([])
        axdendro.set_yticks([])
        index = Z['leaves']
        d = d[index,:]
        d = d[:,index]
        ## Plot
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        im = axmatrix.imshow(d, aspect=1, origin='lower', norm=colors.PowerNorm(gamma=0.2), cmap='Reds')
        _ = axmatrix.set_xticks(np.arange(len(self.cnamelist)))
        _ = axmatrix.set_yticks(np.arange(len(self.cnamelist)))
        _ = axmatrix.set_xticklabels([self.cnamelist[i] for i in index], rotation=90)
        _ = axmatrix.set_yticklabels([self.cnamelist[i] for i in index])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        cbr = plt.colorbar(im, cax=axcolor)
        cbr.set_ticks([0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        fname = os.path.join(self.plotdir, 'ChromosomalProximityIndex.pdf')
        fig.savefig(fname)
        plt.close(fig)







if __name__ == '__main__':
    print
    print '**********************************************'
    print 'Welcome to ChromaSEP test suite!'
    print '**********************************************'
    print
    print 'Initializing ChromaWalker object...'
    # ChromaWalker parameters
    baseres = 50000
    res = 50000
    cnamelist = ['chr%i' % i for i in range(1, 23)] # + ['chrX']
    betalist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    allbeta = False
    norm = 'gfilter_2e5'
    accession = 'GSE63525'
    runlabel = 'GM12878_primary'
    nloop = 0
    meansize = 0.8
    rhomax = 0.8
    bestmeansize = 5.0
    goodLevels = False
    datalist = []
    pars = {
            'rawdatadir': '/home/tanzw/data/hicdata/ProcessedData/',
            #'genomedatadir': '/home/tanzw/data/genomedata/',
            #'rawdatadir': 'asciidata/',
            'genomedatadir': 'asciidata/',
            'genomeref': 'hg19',
            'rundir': 'rundata-%s/' % ('%s-%s' % (accession, runlabel)),
            'accession': accession,
            'runlabel': runlabel,
            'tsetdatadir': 'rundata-%s/TargetsetOptimization/' %
                            ('%s-%s' % (accession, runlabel)),
            'tsetdataprefix': 'Full-ConstructMC',
            'reportdir': 'reports/',
            'baseres': baseres,
            'res': res,
            'norm': norm,
            'meansize': meansize,
            'cnamelist': cnamelist,
            'betalist': betalist,
            'rhomax': rhomax,
            'nloop': nloop
            }
    epigenpars = {
                  #'epigendatadir': 'epigen',                # Sample local files
                  'epigendatadir': 'epigenomic-tracks',
                  'cellLine': runlabel.split('_')[0],
                  'binsize': res,
                  'datalist': datalist
                 }
    ############################################
    ### Create CW instance
    cw = ChromaWalker(pars, epigenpars=epigenpars)
    ###############################################
    # Embedding-specific parameters: Do not change
    embedargs = {
        'radius_exp': 0.6,                  # Partition size scaling exponent
        'radius_factor': 0.025,             # Partition size scaling factor (deprecated)
        'normmode': 2,                      # Normalization mode (deprecated)
        'tri_ineq': 'V2',                   # Method for enforcing triangle inequality in distances
        'alpha': sys.float_info.epsilon,    # Regularization constant (only in embedding, for dealing with zero-interaction pairs)
        'beta': 1.0,                        # Thermal parameter equivalent (only in embedding)
        'gamma': 1.0,                       # Thermal parameter equivalent (only in embedding)
        'delta': 3.0,                       # Exponential decay length scale (deprecated)
        'limitDist': 'None',                # Set a strict distance cutoff on d_ij
        'sizeScale': 20.0,                  # Scale factor for reconstruction: radius of gyration in a.u.
        'takePosEvals': False               # Force reconstruction to silently ignore all negative eigenvalues in distance-geometry embedding
    }
    reconpars = {
        'res': pars['res'],                                 # Hi-C data resolution
        'cnamelist': ['chr'+str(i) for i in range(1, 23)],  # Removing chrX
        'basebeta': 7.0,                                    # Highest common beta
        'rhomax': 0.8,                                      # The highest acceptable value for rho
        'reconlabel': 'GM12878_primary-noX-noresample-0',   # Directory name used to store
        'sampleninterval': 100,                             # Interval of N at which to sample minimum rho
        # Resampling
        'resample': False,                                  # Turn on resampling of interaction counts
        'resample_ratio': 0.9,                              # Reduce interaction counts by this ratio
        'fijnorm': 'gfilter_2e5',                           # Interaction normalization used in ChromaWalker
        # Normalization
        'norm': 'SCPNs',                                    # SCPNs used in paper
        # Reconstruction
        'embedargs': embedargs,
        'nalphasamples': 10,                               # Number of times to sample a given alpha value
        'alphavals': [0.01, 0.02, 0.05, 0.1, 0.2],          # Interaction noise parameter alpha
        'nSEPpoints': 3000,                                 # (Approximate) number of points to sample per SEP genome
        # Directories
        'datadir': 'ReconData',
        'plotdir': 'ReconPlots',
    }
    ############################################
    ### Create CS instance
    print 'Initializing ChromaSEP object...'
    cs = ChromaSEP(reconpars, cw)
    # Get genome-scale hierarchy
    cs.getGlobalHierarchy()
    cs.getReportsHierarchy()
    # Get base interaction matrices and base reconstruction for alignment
    cs.getBaseFab()
    #plt.ion()
    #for n in cs.basefabs:
        #plt.figure()
        #img = plt.imshow(cs.basefabs[n] ** 0.2, cmap='afmhot_r')
        #plt.colorbar(img)
        #plt.title('N=%i' % n)
        #plt.show()
    #_ = raw_input('...:')
    cs.getBaseRecon()      # Perform partition-level reconstructions at each N without ensemble sampling
                           # - These are used to align SEP reconstructions within and between cases of (N, alpha)
    cs.getBaseReconVTF()   # Dump partition-level reconstruction at alpha=0.0 to VTF format
                           # - VMD-native format, chromosomes labelled by SegName
    # Get partition-level reconstruction
    cs.getSEPRecon()       # Perform SEP reconstructions, properly scaled and aligned
    cs.getSEPReconVTF()    # Produce .vtf files for visualization on VMD
    cs.getReportsSEP()     # Get miscellaneous reports on the highest-N reconstructions at the highest alpha



