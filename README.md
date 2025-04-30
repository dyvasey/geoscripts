# Geoscripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dyvasey/geoscripts/HEAD) [![DOI](https://zenodo.org/badge/334745753.svg)](https://zenodo.org/badge/latestdoi/334745753)

## About

Geoscripts is a Python package designed to support geologic data analysis and figure construction. This is largely a personal collection of scripts used for my own academic publications that I have decided to package for my own re-use, to facilitate review of my manuscripts, and ultimately for others to use. 

## Installation

Geoscripts is still in very early stages of package development. It can be installed via `pip` by cloning this repository, navigating into the main directory, and running `pip install`:
```
git clone https://github.com/dyvasey/geoscripts.git
cd geoscripts
pip install .
```

Geoscripts can also be run in a Binder environment using the Binder badge at the top of this README.

## Known Issues

Geoscripts is still in very early stages of development. Most significantly, it currently lacks a test suite and workflow, as well as documentation external to the docstrings within the source code. Thus, the code cannot be guaranteed to run on any specific platform or version of Python.

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

Vasey, D.A., Cowgill, E., and Cooper, K.M., 2021, A Preliminary Framework for Magmatism in Modern Continental Back-Arc Basins and Its Application to the Triassic-Jurassic Tectonic Evolution of the Caucasus: _Geochemistry, Geophysics, Geosystems_, v. 22, e2020GC009490, doi:10.1029/2020GC009490.

Vasey, D.A., Garcia, L., Cowgill, E., Trexler, C.C. and Godoladze, T., 2024, Episodic evolution of a protracted convergent margin revealed by detrital zircon geochronology in the Greater Caucasus: _Basin Research_, v. 36, no. 1, e12825, doi:10.1111/bre.12825

Vasey, D.A., Cowgill, E., VanTongeren, J.A., and Anderson, C.O., accepted, Relict Back-Arc Basin Crustal Structure in the Western Greater Caucasus, Georgia: _Geochemistry, Geophysics, Geosystems_.





