<div align="left">
  <a href="https://github.com/kesslerlib/kessler"> <img height="120px" src="docs/kessler_logo.png"></a>
</div>

-----------------------------------------
[![Build Status](https://github.com/kesslerlib/kessler/workflows/build/badge.svg)](https://github.com/kesslerlib/kessler/actions)

Kessler is a Python package for simulation-based inference and machine learning for space collision avoidance and assessment. It is named in honor of NASA scientist [Donald J. Kessler](https://en.wikipedia.org/wiki/Donald_J._Kessler) known for proposing the [Kessler syndrome](https://en.wikipedia.org/wiki/Kessler_syndrome).

Developed by the [FDL Europe](https://fdleurope.org/) Constellations team in collaboration with [European Space Operations Centre (ESOC)](http://www.esa.int/esoc) of the [European Space Agency (ESA)](http://www.esa.int).


### 0. Prerequisites

- Python 3.6, 3.7 or 3.8.

Note: Python 3.9 is currently not supported due to one of our dependencies (pykep) not supporting it. You might get things working with Python 3.9 if you install [pykep using conda](https://anaconda.org/conda-forge/pykep) (not tested).

### 1. How to install

To install kessler, do the following:

```
git clone https://github.com/kesslerlib/kessler.git
cd kessler
pip install -e .
```

### 2. Get started

To get started, follow the Jupyter notebooks in the [notebooks](https://github.com/kesslerlib/kessler/-/tree/master/notebooks) folder.


### Optional: how to run it in Docker

#### Build the Docker image

In the root folder of this repository, run:
```
docker build -t kessler .
```

#### Run Jupyter inside Docker

If you're using Linux:
```
docker run --rm -it -v $PWD:/workspace --net=host kessler jupyter notebook --allow-root
```

If you're using MacOS:
```
docker run --rm -it -v $PWD:/workspace -p 8888:8888 kessler jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
