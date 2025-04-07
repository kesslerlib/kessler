Kessler Reference Documentation
================================

Kessler is a Python package for machine learning applied to spacecraft collision avoidance. It provides functionalities to import, export, analyze, and plot conjunction data messages (CDMs) in their standard format and predict the evolution of satellite conjunction events based on explainable machine learning models. 

The package comprises a Deep Learning module, where a Bayesian recurrent neural network can be trained with existing collections of CDM data and then deployed in order to predict the contents of future CDMs received up to now, with associated uncertainty estimates about all predictions.

Kessler also includes a novel generative model of conjunction events and CDM sequences implemented using probabilistic programming and simulating the CDM generation process.

For more details on the model and results, check out our publications listed in the README of the repository. 

The authors are [Giacomo Acciarini](https://www.esa.int/gsp/ACT/team/giacomo_acciarini/), [Atılım Güneş Baydin](https://gbaydin.github.io/), Francesco Pinto, and the FDL Europe Constellation team.


```{toctree}
:maxdepth: 1
:caption: Getting Started

install
capabilities
credits
```

```{toctree}
:maxdepth: 1
:caption: Contents

tutorials
api
```

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

