Kessler Reference Documentation
================================

Kessler is a Python package for machine learning applied to spacecraft collision avoidance. It provides functionalities to import, export, analyze, and plot conjunction data messages (CDMs) in their standard format and predict the evolution of satellite conjunction events based on explainable machine learning models. 

The package comprises a Deep Learning module, where a Bayesian recurrent neural network can be trained with existing collections of CDM data and then deployed in order to predict the contents of future CDMs received up to now, with associated uncertainty estimates about all predictions.

Kessler also includes a novel generative model of conjunction events and CDM sequences implemented using probabilistic programming and simulating the CDM generation process.

For more details on the model and results, check out our publications listed in the README of the repository. 

The authors are [Giacomo Acciarini](https://www.esa.int/gsp/ACT/team/giacomo_acciarini/), [Atılım Güneş Baydin](https://gbaydin.github.io/), [Dario Izzo](https://www.esa.int/gsp/ACT/team/dario_izzo/). The main developer is Giacomo Acciarini (giacomo.acciarini@gmail.com).

   :maxdepth: 1
   :caption: Getting started

   install.rst
   capabilities
   credits


   :maxdepth: 1
   :caption: Tutorials

   notebooks/basics
   notebooks/cdms_analysis_and_plotting
   notebooks/LSTM_training
   notebooks/probabilistic_programming_module


   :maxdepth: 1
   :caption: API documentation

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

