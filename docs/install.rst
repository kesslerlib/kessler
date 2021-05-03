Installation
============

Introduction
------------

Currently supported Python versions:

- Python 3.6, 3.7, 3.8.

Note: Python 3.9 is currently not supported due to one of our dependencies (pykep) not supporting it. You might get things working with Python 3.9 if you install `pykep using conda <https://anaconda.org/conda-forge/pykep>`_ (not tested).

Dependencies
------------

It has the following **mandatory** dependencies:

* `Numpy <https://numpy.org/>`_ 
* `Matplotlib <https://matplotlib.org/>`_
* `PyTorch <https://pytorch.org/>`_ (version >=1.5.1)
* `Pykep <https://esa.github.io/pykep/>`_ (version >=2.5)
* `Skyfield <https://rhodesmill.org/skyfield/>`_ (version >=1.26)
* `Pyprob <https://github.com/pyprob/pyprob>`_
* `Pandas <https://pandas.pydata.org/>`_ 

How to install
--------------

To install kessler, do the following:

.. code-block:: console
   
   $ git clone https://github.com/kesslerlib/kessler.git
   $ cd kessler
   $ pip install -e .

Docker (Optional)
-----------------

Build the Docker image:

.. code-block:: console

   $ docker build -t kessler .


You might want to use Jupyter inside Docker.

If you are using Linux:

.. code-block:: console

   $ docker run --rm -it -v $PWD:/workspace --net=host kessler jupyter notebook --allow-root


If you are using MacOS:

.. code-block:: console

   $ docker run --rm -it -v $PWD:/workspace -p 8888:8888 kessler jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

