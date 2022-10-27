# Geoscripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/334745753.svg)](https://zenodo.org/badge/latestdoi/334745753)[![Anaconda-Server Badge](https://anaconda.org/davasey/geoscripts/badges/version.svg)](https://anaconda.org/davasey/geoscripts)

## About

Geoscripts is a Python package designed to support geologic data analysis and figure construction. This is largely a personal collection of scripts used for my own academic publications that I have decided to package for my own re-use, to facilitate review of my manuscripts, and ultimately for others to use. 

## Installation

Geoscripts is still in very early stages of package development. It can currently be installed via conda from my [Anaconda channel](https://anaconda.org/davasey/geoscripts) using the following commands within a conda environment:

`conda install -c davasey geoscripts`

Installation via pip is currently not supported due to the dependency Cartopy, which requires additional compilers, as discussed [here](https://scitools.org.uk/cartopy/docs/latest/installing.html).

Geoscripts can also be run in a Binder environment using the Binder badge at the top of this README.

## Known Issues

Geoscripts is still in very early stages of development. Most significantly, it currently lacks a test suite and workflow, as well as documentation external to the docstrings within the source code. Thus, the code cannot be guaranteed to run on any specific platform or version of Python and is only supported for Python 3.10 at the moment.

## Dependencies

Geoscripts currently has a long list of specific dependencies; a goal for future development is to reduce this list, perhaps by moving some functionality into other projects:
* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Shapely
* Cartopy
* statsmodels
* GeoPandas
* mpltern
* Pyrolite

## Publications Using Geoscripts

Vasey, D.A., Cowgill, E., and Cooper, K.M., 2021, A Preliminary Framework for Magmatism in Modern Continental Back-Arc Basins and Its Application to the Triassic-Jurassic Tectonic Evolution of the Caucasus: _Geochemistry, Geophysics, Geosystems_, v. 22, p. e2020GC009490, doi:https://doi.org/10.1029/2020GC009490.





