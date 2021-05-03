Capabilities
============

Overview
--------

Kessler is an open-source Python package that currently includes Bayesian ML and probabilistic programming components. The library currently provides the following key capabiolities:

* Functionality to import and export Conjunction Data Messages (CDM) data, using either the CDM standard format or databases that can be connected to Kessler through an API based on pandas ``DataFrame`` objects, and grouping CDMs into ``Events`` objects representing conjunctions.

* Plotting code to visualize event evolution of existing CDMs or predicted ones.

* A ML module that currently implements a stack of Bayesian long short-term memory (LSTM) neural networks that can be used to train with user's private collection of CDM data.

* A probabilistic programming module simulating conjunction events and CDM generation process. This can be useful both for performing event analysis using Bayesian inference or for generating synthetic CDM datasets sampled from this probabilistic generative model.


 
