CRISP
==============================
[//]: # (Badges)
[![Documentation Status](https://readthedocs.org/projects/crisp-host/badge/?version=latest)](https://crisp-host.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/crisp-ase.svg)](https://badge.fury.io/py/crisp-ase)
[![Python Version](https://img.shields.io/pypi/pyversions/crisp-ase.svg)](https://pypi.org/project/crisp-ase/)
[![License](https://img.shields.io/github/license/Indranil17/CRISP_HOST.svg)](https://github.com/Indranil17/CRISP_HOST/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI.svg)](https://doi.org/YOUR_DOI)


<img src="https://github.com/Indranil17/CRISP_HOST/blob/main/crisp_logo2.png" width="300">

# CRISP: Enhancing ASE Workflows with Advanced Molecular Simulation Post-Processing

## What is CRISP?
CRISP is a post-simulation analysis package built on the Atomic Simulation Environment (ASE). It is designed for efficient and insightful analysis of molecular dynamics (MD) and other simulations, enabling in-depth exploration with just a few lines of code, including powerful visualisation options.

## Features
- **User-friendly**: Optimised for ease of use with detailed examples and extensive outputs for nuanced data analysis.
- **Highly parallelised**: Utilises parallelisation techniques that scale linearly with the number of CPU cores, allowing for fast analysis of large systems and long simulations on high-performance computing clusters.
- **Interactive Visualisation**: HTML outputs to interact with, which can help to see structural changes.  
- **Integration**: ASE, scikit-learn and other Python packages.  

## Analysis Toolkit Highlights 

### 1. Cluster Comprehension
- Perform in-depth clustering analysis in molecular dynamics or Monte Carlo simulations using advanced algorithms like DBSCAN.
- Works with both periodic and non-periodic systems.
- Identify, visualise, and track distinct atom clusters to gain insights into unbiased clustering of selected atoms.

### 2. Customizable Radial Distribution Functions
- Compute and plot partial radial distribution functions (PRDF) for selected atoms or atom types.
- Easily analyse radial relationships between atoms with periodic boundary conditions.

### 3. Mean Square Displacement (MSD)
- Quantify atomic motion over time using MSD calculations, providing key insights into diffusion and dynamics.
- Customise analysis by selecting specific atom indices to focus on particular subsets of atoms.

### 4. Hydrogen Bond Analysis
- Identify and analyse hydrogen bonds with a single line of code.
- Customise hydrogen bond parameters or atom indices for detailed and specific analysis.
- Track structural parameters to understand the nature and stability of hydrogen bonds in your system.

### 5. Coordination Analysis
- Compute average coordination numbers for specific atom types with customizable cutoffs.
- Analyse contact times of selected atom types to study dynamic behaviour efficiently.

### 6. Error Analysis
- Accurately estimate the error of any computed property using statistical techniques.
- Choose between autocorrelation function or block averaging to calculate the error of the mean, improving result reliability.
- Assess simulation convergence by analysing vector or scalar properties like atomic positions or energy.

### 7. Efficient and Robust Sampling
- Sample structures using Furthest Point Sampling (FPS) with SOAP descriptors.
- Efficiently subsample large databases or simulations by selecting the most diverse structures while avoiding redundancy.

### 7. Volumetric Density
- Create 3D volumetric density maps to visualise the spatial distribution of atoms throughout a trajectory
- Understanding diffusion pathways, adsorption sites etc.  



# Requirements
This package is built around the ASE (Atomic Simulation Environment) and thus requires the installation of ASE, available at: [ASE Installation Guide](https://wiki.fysik.dtu.dk/ase/install.html).

The clustering uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implemented by scikit-learn, available at: [scikit-learn Installation](https://scikit-learn.org/stable/install.html).

For interactive 3D plots of clustering, CRISP utilises the seaborn package. Ensure you have seaborn installed by following the instructions at: [Seaborn Installation](https://seaborn.pydata.org/installing.html).

# Installation
- Recommended: install from PyPI (tested on Python 3.9–3.11 on Linux/macOS/Windows)
	```
	pip install crisp-ase==1.1.2
	```

- From a local checkout (same supported Python versions)
	```
	git clone https://github.com/Indranil17/CRISP.git
	cd CRISP
	python -m pip install -U pip setuptools wheel
	python -m pip install .
	```

- Dependencies (if installing manually) 
	```
	python -m pip install \
		ase>=3.23.0 \
		scikit-learn>=1.4.2 \
		seaborn>=0.12.2 \
		joblib>=1.2.0 \
		fpsample>=0.3.3,<1.0 \
		dscribe>=2.0.0 \
		statsmodels>=0.14.0 \
		pandas>=2.0.3 \
		plotly>=5.9.0 \
		networkx>=3.1
	```

# Current Modules

To learn how to use each of the modules, please visit [examples](https://github.com/Indranil17/CRISP_HOST/tree/main/example).
Inside the example folder, check the [Notebook](https://github.com/Indranil17/CRISP_HOST/blob/main/example/Introductory_Tutorial/CRISP_Introdcutory_Tutorial.ipynb) to see them in use and also check the other folders of the example folder to see the outputs.
It has two sub-packages and ten modules.  
Please see the package's UML map below. 
 
<img src="https://github.com/Indranil17/CRISP_HOST/blob/main/crisp_map.png" width="800">



# Acknowledgments
The package is from the (Nano)Materials modelling group, at Charles University.  
Email: lukas.grajciar@natur.cuni.cz  

<img src="https://github.com/Indranil17/CRISP_HOST/blob/main/group_logo.png" width="100">

### Contributors
If there is something you would like to see added to this package or if you would like to contribute,  
please email us at sahai@natur.cuni.cz or daniel.willimetz@natur.cuni.cz

