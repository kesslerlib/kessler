Loading CDMs from `Kelvins Challenge <https://kelvins.esa.int/collision-avoidance-challenge/>`_ dataset
=======================================================================================================

In this tutorial, we show the case in which the data to be loaded comes from the Kelvins competition (i.e., https://kelvins.esa.int/collision-avoidance-challenge/data/): a collision avoidance challenge organized by ESA in 2019.

For this purpose, we built a specific converter that takes care of the conversion from the Kelvins format to standard CDM format.
First, we perform the relevant imports:

.. code-block:: python

   import kessler
   from kessler.data import kelvins_to_event_dataset

Then, we proceed in converting the Kelvins dataset as an ``EventDataset`` objetc. In the following example, we leverage two extra entries (i.e., ``drop_features`` and ``num_events``) to exclude certain features when importing, and to only import a limited number of events (in this case 1000).

.. code-block:: python
   
   file_name='path/to/train_data.csv'
   events=kelvins_to_event_dataset(file_name, drop_features=['c_rcs_estimate', 't_rcs_estimate'], num_events=1000)

The output will show the number of CDMs and events loaded, as they progress.
