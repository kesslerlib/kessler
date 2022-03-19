Kessler Reference Documentation
================================

Kessler is a Python package for machine learning applied to spacecraft collision avoidance. It provides functionalities to import, export, analyze, and plot conjunction data messages (CDMs) in their standard format and predict the evolution of satellite conjunction events based on explainable machine learning models. 

The package comprises a Deep Learning module, where a Bayesian recurrent neural network can be trained with existing collections of CDM data and then deployed in order to predict the contents of future CDMs received up to now, with associated uncertainty estimates about all predictions.

Kessler also includes a novel generative model of conjunction events and CDM sequences implemented using probabilistic programming and simulating the CDM generation process, which we will soon release to the public: stay tuned!

The documentation is currently a work in progress.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   install
   capabilities
   credits


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/basics
   tutorials/cdms_analysis_and_plotting
   tutorials/LSTM_training
   tutorials/probabilistic_programming_module


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   kessler package <_autosummary/kessler>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
