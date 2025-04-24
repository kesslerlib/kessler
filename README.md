# Kessler

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kesslerlib/kessler">
    <img src="docs/_static/kessler_logo.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Kessler
    <br />
    <a href="https://kesslerlib.github.io/kessler/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kesslerlib/kessler/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/kesslerlib/kessler/issues/new/choose">Request feature</a>
  </p>
</p>

-----------------------------------------
[![Build Status](https://github.com/kesslerlib/kessler/workflows/build/badge.svg)](https://github.com/kesslerlib/kessler/actions)
[![codecov](https://codecov.io/gh/kesslerlib/kessler/branch/master/graph/badge.svg?token=EQ9CLXD909)](https://codecov.io/gh/kesslerlib/kessler)

Kessler is a Python package for simulation-based inference and machine learning for space collision avoidance and assessment. It is named in honor of NASA scientist [Donald J. Kessler](https://en.wikipedia.org/wiki/Donald_J._Kessler) known for his studies regarding [space debris](https://en.wikipedia.org/wiki/Space_debris) and proposing the [Kessler syndrome](https://en.wikipedia.org/wiki/Kessler_syndrome).

Initially developed by the [FDL Europe](https://fdleurope.org/) Constellations team in collaboration with [European Space Operations Centre (ESOC)](http://www.esa.int/esoc) of the [European Space Agency (ESA)](http://www.esa.int).

## Documentation and roadmap

To get started, follow the [documentation](https://kesslerlib.github.io/kessler/) examples.

## Authors

Kessler was initiated by the Constellations team at the Frontier Development Lab (FDL) Europe 2020, a public–private partnership between the European Space Agency (ESA), Trillium Technologies, and University of Oxford. The main developer is [Giacomo Acciarini](https://www.esa.int/gsp/ACT/team/giacomo_acciarini/).

Constellations team members: Giacomo Acciarini, Francesco Pinto, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, Atılım Güneş Baydin

## License

Kessler is distributed under the GNU General Public License version 3. Get in touch with the authors for other licensing options.

## More info and how to cite

If you use `kessler`, we would be grateful if you could star the repository and/or cite our work.
If you would like to learn more about or cite the techniques `kessler` uses, please see the following papers:

* Giacomo Acciarini, Nicola Baresi, Christopher Bridges, Leonard Felicetti, Stephen Hobbs, Atılım Güneş Baydin. 2023. [“Observation Strategies and Megaconstellations Impact on Current LEO Population.”](https://conference.sdo.esoc.esa.int/proceedings/neosst2/paper/88) In 2nd NEO and Debris Detection Conference.
```
@inproceedings{acciarini-2023-observation,
  title = {Observation Strategies and Megaconstellations Impact on Current LEO Population},
  author = {Acciarini, Giacomo and Baresi, Nicola and Bridges, Christopher and Felicetti, Leonard and Hobbs, Stephen and Baydin, Atılım Güneş},
  booktitle = {2nd NEO and Debris Detection Conference},
  year = {2023}
}
```
* Giacomo Acciarini, Francesco Pinto, Francesca Letizia, José A. Martinez-Heras, Klaus Merz, Christopher Bridges, and Atılım Güneş Baydin. 2021. [“Kessler: a Machine Learning Library for Spacecraft Collision Avoidance.”](https://conference.sdo.esoc.esa.int/proceedings/sdc8/paper/226) In 8th European Conference on Space Debris.
```bibtex
@inproceedings{acciarini-2020-kessler,
  title = {Kessler: a Machine Learning Library for Spacecraft Collision Avoidance},
  author = {Acciarini, Giacomo and Pinto, Francesco and Letizia, Francesca and Martinez-Heras, José A. and Merz, Klaus and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {8th European Conference on Space Debris},
  year = {2021}
}
```
* Francesco Pinto, Giacomo Acciarini, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, and Atılım Güneş Baydin. 2020. “Towards Automated Satellite Conjunction Management with Bayesian Deep Learning.” In AI for Earth Sciences Workshop at NeurIPS 2020, Vancouver, Canada. [arXiv:2012.12450](https://arxiv.org/abs/2012.12450)
```bibtex
@inproceedings{pinto-2020-automated,
  title = {Towards Automated Satellite Conjunction Management with Bayesian Deep Learning},
  author = {Pinto, Francesco and Acciarini, Giacomo and Metz, Sascha and Boufelja, Sarah and Kaczmarek, Sylvester and Merz, Klaus and Martinez-Heras, José A. and Letizia, Francesca and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {AI for Earth Sciences Workshop at NeurIPS 2020, Vancouver, Canada},
  year = {2020}
}
```
* Giacomo Acciarini, Francesco Pinto, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, and Atılım Güneş Baydin. 2020. “Spacecraft Collision Risk Assessment with Probabilistic Programming.” In Third Workshop on Machine Learning and the Physical Sciences (NeurIPS 2020), Vancouver, Canada. [arXiv:2012.10260](https://arxiv.org/abs/2012.10260)
```bibtex
@inproceedings{acciarini-2020-spacecraft,
  title = {Spacecraft Collision Risk Assessment with Probabilistic Programming},
  author = {Acciarini, Giacomo and Pinto, Francesco and Metz, Sascha and Boufelja, Sarah and Kaczmarek, Sylvester and Merz, Klaus and Martinez-Heras, José A. and Letizia, Francesca and Bridges, Christopher and Baydin, Atılım Güneş},
  booktitle = {Third Workshop on Machine Learning and the Physical Sciences (NeurIPS 2020), Vancouver, Canada},
  year = {2020}
}
```

## Installation

To install `kessler` locally, you can do the following:

```
git clone https://github.com/kesslerlib/kessler.git
cd kessler
pip install -e .
```
