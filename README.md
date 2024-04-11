# ChromaSEP: Stochastic Embedding Procedure for genome ensemble 3D reconstructions using Hi-C data #

### Introduction ###

* This is a Python package developed for reconstructing chromatin 3-dimensional structural ensembles captured Hi-C data experiments.
* The pipeline is based on a Markov-State Model analysis of the chromatin structural hierarchy (see Tan, Z. W.; Guarnera, E.; Berezovsky, I. N. Exploring Chromatin Hierarchical Organization via Markov State Modelling. PLoS Computational Biology 2018, 14 (12). https://doi.org/10.1371/journal.pcbi.1006686.
), followed by spatial embedding of interaction data using classical distance geometry methods
* This package is associated with the following paper: Guarnera, E.; Tan, Z. W.; Berezovsky, I. N. Three-Dimensional Chromatin Ensemble Reconstruction via Stochastic Embedding. Structure 2021, 29 (6), 622-634.e3. https://doi.org/10.1016/j.str.2021.01.008
* This package was developed in Berezovsky Lab at the Bioinformatics Institute ASTAR (Agency for Science, Technology, and Research) Singapore.
* This package is distributed under the MIT Licence.

### Setting up ###

BLAS/LAPACK for optimized linear algebra computations

* Linear algebra computations employed in the package appear to perform much faster when NumPy/SciPy Python libraries are installed after BLAS/LAPACK.
* For Debian systems, run `sudo apt-get install libblas-dev liblapack-dev`.
* For Red Hat-based systems, run `sudo yum install lapack-devel blas-devel`.


Python requirements

* Python version: 2.7 (this package has been tested for versions 2.7.11 to 2.7.18)
* Required libraries
	* SciPy stack (NumPy, SciPy, Matplotlib, Pandas)
	* rmsd (https://anaconda.org/conda-forge/rmsd)
* Helpful libraries
	* Jupyter Lab (https://anaconda.org/conda-forge/jupyterlab)
* The easiest way to set up the Python 2.7 environment and required libraries is to use anaconda/miniconda (`conda create -n py2 python=2.7`, `conda install -c anaconda numpy scipy`, etc.).
* Jupyter Lab provides a convenient way to visually track/organize the computational pipeline.

Visualizing chromatin ensembles

* A useful tool for visualizing chromatin ensembles generated using the SEP routine is VMD (https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD), 
  a molecular visualization toolkit that is free for academic use.
* The output VTF format for SEP reconstructions is catered for visualization on VMD.

Downloading the package

* The easiest way is to use `git clone`.
	* Note that this repository contains both the ChromaWalker and ChromaSEP packages.


### How to use the package ###

The ChromaSEP package uses partitioning data obtained from the Markov State Model analysis implemented in the ChromaWalker package, with each single-chromosome partitioning level 
characterized by a metastability index $\rho$. Part of the implemented pipeline involves merging different combinations of single-chromosome partitionings to form a whole-genome 
partitioning, and then optimizing whole-genome partitionings by minimizing $\rho$ for any fixed number of partitions $N$. The ChromaSEP package was designed to work in the same 
directory as the ChromaWalker package, thos users executing the full pipeline do not have to worry about directory structure and naming conventions.

We assume that the user has completed hub set optimization analysis at the highest common permissible value of beta (`beta = basebeta`), through the following procedure:

```
cw = ChromaWalker(pars, epigenpars=epigenpars)
cw.getAllFMCmats()
cw.autoTsetOptimization()
```

The basic workflow for ChromaSEP is documented in the `if '__name__' == '__main__':` block. Look through the definitions of each parameter in the comments provided, and adjust 
them accordingly. Typical users would only need to adjust these: `res, cnamelist, basebeta, reconlabel, nalphasamples, nSEPpoints`.

* Step 0: Create ChromaSEP instance
	* After finalizing input parameters in `reconpars`, create a ChromaSEP instance using `cs = ChromaSEP(reconpars, cw)`. This creates a ChromaSEP data directory `{reconlabel}`.
* Step 1: Optimize global metastability index for different number of partitions $N$
	* `cs.getGlobalHierarchy()` attempts to optimize the $\rho(N)$ curve (Fig 1F in publication) via Monte Carlo sampling.
	* `cs.getReportsHierarchy()` plots the $\rho(N)$ curve to a PDF in `{reconlabel}/ReconPlots/`, and stores data on different levels of structural hierarchy in 
	  `{reconlabel}/ReconData/globalHierarchyData.p`. Users may call `cs.getGlobalHierarchy()` again to further optimize the curve.
* Step 2: Obtain effective interaction matrices at different levels of whole-genome partitioning
	* `cs.getBaseFab()` obtains the non-noisy effective interaction matrices for inter-partition communication (`{reconlabel}/ReconData/baseFabs.p`), based on Hi-C data 
	  processed in ChromaWalker.
* Step 3: Get partition-level reconstructions of unperturbed effective interaction matrix
	* `cs.getBaseRecon()` runs the classical embedding routine on base effective interaction matrices, normalized using the routine specified by `norm`
	  (we suggest sticking to `norm = 'SCPNs'`). The structures obtained (`{reconlabel}/ReconData/baserecons.p`) are used to align structures obtained at different levels of hierarchy.
	* (Optional) `cs.getBaseReconVTF()` outputs base reconstructions (`{reconlabel}/ReconData/baserecons/`) in VTF format for visualization.
* Step 4: Get SEP reconstructions on noisy effective interaction matrices
	* `cs.getSEPRecon()` produces `nalphasamples` partition-level reconstructions of at each noise level `alpha` (`{reconlabel}/ReconData/alpharecons/`), and then samples
	  intra-partition structure to produce full SEP reconstructions ('{reconlabel}/ReconData/SEPrecons/').
	* `cs.getSEPReconVTF()` produces VTF files corresponding to SEP reconstructions (`{reconlabel}/ReconData/SEPrecons/`) for visualization. Each sample is represented as a single frame.
* Step 5: Get structural analysis reports
	* `cs.getReportsSEP()` performs the following characterizations of the obtained SEP reconstructions:
		* Rescaled RMSD bvetween levels of hierarchy, as a function of noise level `alpha`
		* Single-chromosome distance correlations $g(r)$ at the highest `N` and `alpha`
		* Chromosomal proximity index matrix at the highest `N` and `alpha`


### Skipping ChromaWalker analysis ###

Users may choose to skip the native partitioning analysis and begin with their own (hierarchical) partitioning of the whole genome by using a dummy ChromaWalker instance, and 
supplying the following Python pickles in the working directory, and starting from Step 3:

* `{reconlabel}/ReconData/globalHierarchyData.p`
	* This is a Python dict with the following keys:
		* globalNs: a list of the number of partitions at each level of hierarchy. (e.g., `[100, 200, 300, ...]`)
		* nchoices: a list of Python dicts - number of partitions for each chromosome, at each level of hierarchy. (e.g., `[{'chr1': 5, 'chr2': 3, ...}, {'chr1': 20, ...}, {...}]`)
		* partitiondata: a Python dict with keys equal to the number of partitions at each selected level of hierarchy, and corresponding values are lists demarcating the 
		  boundaries of each partition. For example, `{100: [('chr1', (1, 200)), ('chr1', (200, 350)), ('chr1', (400, 553)), ..., ('chr2', (0, 100)), ...], 200: [...]}` indicates that at 
		  the N=200 level of hierarchy, partitions span the genomic ranges chr1:[1 * res]-[200 * res], chr1:[200 * res]-[350 * res], chr1:[400 * res]-[553 * res], ..., 
		  chr2:[0 * res]-[100 * res], ..., where `res`, the resolution of the Hi-C data, is a parameter in the `pars` dict used to create the ChromaSEP instance.
* `{reconlabel}/ReconData/basefabs.p`
	* This is a Python dict with keys equal to the number of partitions at each selected level of hierarchy, and the corresponding values are square matrices of the effective 
	  interaction matrices (numerically equivalent to coarse-grained interaction matrices).

### Contact ###

* If you encounter any difficulties with installing or running the package, please contact me (Zhen Wah) at tanzw[AT]bii.a-star.edu.sg
