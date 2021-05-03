Load CDMs from ``.kvn``
=======================

In this tutorial, we show how to load CDMs from ``.kvn`` format.

First, the CDMs in ``.kvn`` format need to be placed inside the ``path_to_cdms_folder``, for correctly loading the data. Furthermore, the code expects the CDMs in the folder to have file names grouped by: individual event and the CDM sequence in each event.

For instance, if we have to load two events with 3 and 2 CDMs each, we might then have file names in the following format:

* ``event_1_01.cdm.kvn.txt``
* ``event_1_02.cdm.kvn.txt``
* ``event_1_03.cdm.kvn.txt``
* ``event_2_01.cdm.kvn.txt``
* ``event_2_02.cdm.kvn.txt``

For loading the events, we first need to do the relevant imports:

.. code-block:: python

   import kessler
   from kessler import EventDataset

We can then proceed in creating the ``EventDataset`` object:

.. code-block:: python
   
   path_to_cdms_folder='path/to/folder'
   events=EventDataset(path_to_cdms_folder)

A message appears confirming that the loading has happened, with the number of CDMs and events.
