
import numpy as np
from Bio.PDB import PDBParser


# Functions for protein structure and coordinate analysis


def getPDB_Calpha(fname):
  """
  Extract coordinates of Calpha atoms from protein PDB file.
  Ignores chains with no Calpha atoms (non-protein chains).
  Names chains as '1', '2', etc.
  Returns: coords (N, 3), chains (N,)
  """
  CA_coordinates = []
  p = PDBParser()
  structure = p.get_structure('name', fname)
  for model in structure:
    for chain in model:
      CA_coordinates.append([])
      for residue in chain:
        if 'CA' not in residue:
          continue
        CA_coordinates[-1].append(np.array(residue['CA'].get_coord(),
                            dtype='float64'))
  # Take only chains with at least 1 CA
  coords = []
  chains = []
  thischain = 0
  for chain in CA_coordinates:
    if len(chain) > 0:
      thischain += 1
      coords.extend(chain)
      chains.extend([str(thischain)] * len(chain))
  return np.array(coords), np.array(chains)


def getPDB_Calpha_labels(fname):
  """
  Extract coordinates of Calpha atoms from protein PDB file.
  Ignores chains with no Calpha atoms (non-protein chains).
  Names chains as '1', '2', etc.
  Returns: coords (N, 3), chains (N,), chainids, resids
  """
  CA_coordinates = []
  cidlist = []
  residlist = []
  p = PDBParser()
  structure = p.get_structure('name', fname)
  for model in structure:
    for chain in model:
      thischain = chain.id
      CA_coordinates.append([])
      for residue in chain:
        if 'CA' not in residue:
          continue
        CA_coordinates[-1].append(np.array(residue['CA'].get_coord(),
                            dtype='float64'))
        cidlist.append(thischain)
        residlist.append(residue.get_id()[1])
  # Take only chains with at least 1 CA
  coords = []
  chains = []
  thischain = 0
  for chain in CA_coordinates:
    if len(chain) > 0:
      thischain += 1
      coords.extend(chain)
      chains.extend([str(thischain)] * len(chain))
  return np.array(coords), np.array(chains), \
          np.array(cidlist), np.array(residlist)


def proteinCG(coords, chains, cgfactor):
  """
  Coarse-grain protein into blobs of multiple residues.
  Returns: cgcoords, res2blobmap, cgchains.
  """
  chainsuniq = np.unique(chains)
  cgcoords = []
  cgchains = []
  res2blobmap = np.zeros(len(coords), dtype=int)
  for c in chainsuniq:
    coordsinchain = coords[chains == c]
    n = len(coordsinchain)
    res2blobmapchain = np.zeros(n, dtype=int)
    npts = n / cgfactor + (0 if (n % cgfactor == 0) else 1)
    for i in range(npts):
      cgcoords.append(np.average(coordsinchain[i * cgfactor : (i + 1) * cgfactor],
              axis=0))
      st = i * cgfactor
      en = (i + 1) * cgfactor if n > (i + 1) * cgfactor else n
      res2blobmapchain[st:en] = len(cgcoords) - 1
    res2blobmap[chains == c] = res2blobmapchain
    cgchains.extend([c] * npts)
  cgcoords = np.array(cgcoords)
  cgchains = np.array(cgchains)
  return cgcoords, res2blobmap, cgchains


def proteinCG_2(coords, chains, fij, cgfactor):
  """
  Coarse-grain protein into blobs of multiple residues.
  Returns: cgcoords, res2blobmap, cgchains, cgfij.
  """
  chainsuniq = np.unique(chains)
  cgcoords = []
  cgchains = []
  n1 = len(coords)
  res2blobmap = np.zeros(n1, dtype=int)
  for c in chainsuniq:
    coordsinchain = coords[chains == c]
    n = len(coordsinchain)
    res2blobmapchain = np.zeros(n, dtype=int)
    npts = n / cgfactor + (0 if (n % cgfactor == 0) else 1)
    for i in range(npts):
      cgcoords.append(np.average(coordsinchain[i * cgfactor : (i + 1) * cgfactor],
              axis=0))
      st = i * cgfactor
      en = (i + 1) * cgfactor if n > (i + 1) * cgfactor else n
      res2blobmapchain[st:en] = len(cgcoords) - 1
    res2blobmap[chains == c] = res2blobmapchain
    cgchains.extend([c] * npts)
  cgcoords = np.array(cgcoords)
  cgchains = np.array(cgchains)
  n2 = len(cgcoords)
  faj = np.zeros((n2, n1))
  fab = np.zeros((n2, n2))
  for i in range(n1):
    faj[res2blobmap[i]] += fij[i] / float(np.sum(res2blobmap == (res2blobmap[i])))
  for j in range(n1):
    fab[:, res2blobmap[j]] += faj[:, j] / float(np.sum(res2blobmap == (res2blobmap[j])))
  fab -= np.diag(np.diag(fab))
  return cgcoords, res2blobmap, cgchains, fab


def addSegmentIDs(fnamein, fnameout, segids, chids, resids):
  """
  Add segment IDs to PDB file
  """
  chresids = ['%s%i' % (ch, res) for ch, res in zip(chids, resids)]
  with open(fnameout, 'w') as fout:
    with open(fnamein, 'r') as fin:
      for line in fin:
        if line[:6] == 'HETATM':
          newline = line[:66] + ('%10s' % 'ZZZ') + line[76:]
          fout.write(newline)
        elif line[:6] != 'ATOM  ':
          fout.write(line)
        else:
          thisch, thisres = line[21], int(line[22:26])
          thischres = '%s%i' % (thisch, thisres)
          if thischres not in chresids:
            newline = line[:66] + ('%10s' % 'ZZZ') + line[76:]
            fout.write(newline)
          else:
            thisseg = segids[chresids.index(thischres)]
            pass
            newline = line[:66] + ('%10s' % thisseg) + line[76:]
            fout.write(newline)


def _append_fullPDB_multiSeg(fnamein, fnameout, segmentlist):
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
    with open(fnamein, 'r') as fin:
      with open(fnameout, 'w') as fout:
        thisres = -9999
        thischain = ''
        thisind = -1
        thismod = -1
        for line in fin:
          if line[:5] == 'MODEL':
            # new model
            thisres = -9999
            thisind = -1
            thismod = int(line[10:]) - 1
            fout.write(line)
          elif line[:6] == 'ENDMDL':
            # end model
            fout.write(line)
          elif line[:4] == 'ATOM':
            # atom
            thislineres = int(line[22:26])
            thislinechain = line[21]
            if thislineres != thisres or thislinechain != thischain:
              thisres = thislineres
              thischain = thislinechain
              thisind += 1
            seg = segmentlist[thismod][thisind]
            fout.write(line[:66] + '%10s' % seg + line[76:])
            pass
          else:
            # Other lines
            fout.write(line)


