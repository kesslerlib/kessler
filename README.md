<div align="left">
  <a href="https://github.com/kesslerlib/kessler"> <img height="120px" src="docs/source/_static/kessler_logo.png"></a>
</div>

-----------------------------------------
[![Build Status](https://github.com/kesslerlib/kessler/workflows/build/badge.svg)](https://github.com/kesslerlib/kessler/actions)
[![Documentation Status](https://readthedocs.org/projects/kessler/badge/?version=latest)](https://kessler.readthedocs.io/en/latest/?badge=latest)

Kessler is a Python package for simulation-based inference and machine learning for space collision avoidance and assessment. It is named in honor of NASA scientist [Donald J. Kessler](https://en.wikipedia.org/wiki/Donald_J._Kessler) known for proposing the [Kessler syndrome](https://en.wikipedia.org/wiki/Kessler_syndrome).

Developed by the [FDL Europe](https://fdleurope.org/) Constellations team in collaboration with [European Space Operations Centre (ESOC)](http://www.esa.int/esoc) of the [European Space Agency (ESA)](http://www.esa.int).

## Documentation and roadmap

To get started, follow the Jupyter notebooks in the [notebooks](https://github.com/kesslerlib/kessler/-/tree/master/notebooks) folder.

The upcoming version of Kessler will inclue the probabilistic programming model of conjunctions, which is excluded from this initial release. 


## Authors

* Giacomo Acciarini, University of Oxford
* Francesco Pinto, University of Oxford
* Francesca Letizia, European Space Agency
* Chris Bridges, University of Surrey
* Atılım Güneş Baydin, University of Oxford

Kessler was initially developed by the Constellations team at the Frontier Development Lab (FDL) Europe 2020, a public–private partnership between the European Space Agency (ESA), Trillium Technologies, and University of Oxford.

Constellations team members: Giacomo Acciarini, Francesco Pinto, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, Atılım Güneş Baydin

## License

Kessler is distributed under the GNU General Public License version 3. Get in touch with the authors for other licensing options.

## More info and how to cite

If you would like to learn more about or cite the techniques Kessler uses, please see the following papers:

* Giacomo Acciarini, Francesco Pinto, Francesca Letizia, José A. Martinez-Heras, Klaus Merz, Christopher Bridges, and Atılım Güneş Baydin. 2021. [“Kessler: a Machine Learning Library for Spacecraft Collision Avoidance.”](https://conference.sdo.esoc.esa.int/proceedings/sdc8/paper/226) In 8th European Conference on Space Debris.
```
@inproceedings{acciarini-2020-kessler,
  title = {Kessler: a Machine Learning Library for Spacecraft Collision Avoidance},
  author = {Acciarini, Giacomo and Pinto, Francesco and Letizia, Francesca and Martinez-Heras, José A. and Merz, Klaus and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {8th European Conference on Space Debris},
  year = {2021}
}
```
* Francesco Pinto, Giacomo Acciarini, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, and Atılım Güneş Baydin. 2020. “Towards Automated Satellite Conjunction Management with Bayesian Deep Learning.” In AI for Earth Sciences Workshop at NeurIPS 2020, Vancouver, Canada. [arXiv:2012.12450](https://arxiv.org/abs/2012.12450)
```
@inproceedings{pinto-2020-automated,
  title = {Towards Automated Satellite Conjunction Management with Bayesian Deep Learning},
  author = {Pinto, Francesco and Acciarini, Giacomo and Metz, Sascha and Boufelja, Sarah and Kaczmarek, Sylvester and Merz, Klaus and Martinez-Heras, José A. and Letizia, Francesca and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {AI for Earth Sciences Workshop at NeurIPS 2020, Vancouver, Canada},
  year = {2020}
}
```
* Giacomo Acciarini, Francesco Pinto, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, and Atılım Güneş Baydin. 2020. “Spacecraft Collision Risk Assessment with Probabilistic Programming.” In Third Workshop on Machine Learning and the Physical Sciences (NeurIPS 2020), Vancouver, Canada. [arXiv:2012.10260](https://arxiv.org/abs/2012.10260)
```
@inproceedings{acciarini-2020-spacecraft,
  title = {Spacecraft Collision Risk Assessment with Probabilistic Programming},
  author = {Acciarini, Giacomo and Pinto, Francesco and Metz, Sascha and Boufelja, Sarah and Kaczmarek, Sylvester and Merz, Klaus and Martinez-Heras, José A. and Letizia, Francesca and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {Third Workshop on Machine Learning and the Physical Sciences (NeurIPS 2020), Vancouver, Canada},
  year = {2020}
}
```

## Installation

### Prerequisites

- Python 3.6, 3.7 or 3.8.

Note: Python 3.9 is currently not supported due to one of our dependencies (pykep) not supporting it. You might get things working with Python 3.9 if you install [pykep using conda](https://anaconda.org/conda-forge/pykep) (not tested).

### How to install

To install kessler, do the following:

```
git clone https://github.com/kesslerlib/kessler.git
cd kessler
pip install -e .
```

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
