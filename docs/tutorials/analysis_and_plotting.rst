Analyze and plot loaded CDMs
============================

In this tutorial, we show how to analyze and plot CDM data.

We assume that data has been loaded (see relevant tutorials on how to do it) and that events are stored in the ``events`` variable. Then, we can first inspect some descriptive statistics and save it to a file:

.. code-block:: python

   import pandas as pd
   events_stats=events.to_dataframe().describe()

Now, we can proceed in plotting events features. First, we start by plotting a single feature of a specific event (e.g. the 31st one):

.. code-block:: python
   
   events[30].plot_feature('OBJECT1_CT_T', file_name='single_feature_single_event.pdf')

But we can also plot multiple features of multiple events in the same figure:

.. code-block:: python

   features=['OBJECT1_CT_T', 'OBJECT2_CT_T']
   events.plot_features(features, file_name='multi_features_multi_events.pdf')

Furthermore, we can leverage a built-in function to plot the covariance matrix evolution of a specific event (all the covariance matrix elements will be plotted in multiple sub-figures):

.. code-block:: python
   
   events[30].plot_uncertainty(file_name='uncertainties_single_event.pdf')

And we can do the same as above, but for all the events:

.. code-block:: python

   events.plot_uncertainty(file_name='uncertainties_multi_events.pdf')

Note that all these plotting methods support an optional ``apply_func`` argument that allows to apply a custom lambda function to the data before plotting.
