Loading CDMs from pandas ``DataFrame`` object
=============================================

In this tutorial, we show how to load CDMs from pandas ``DataFrame`` object.

First we perform the relevant imports:

.. code-block:: python

   import kessler
   import pandas as pd
   from kessler import EventDataset

Then, we create the ``EventDataset`` object, after having uploaded the pandas dataframe and created the ``DataFrame`` object:

.. code-block:: python

   file_name='path/to/file.csv'
   df=pd.read_csv(file_name)
   events=EventDataset.from_pandas(df)
