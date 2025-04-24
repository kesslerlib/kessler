Installation
============

.. _installation_deps:


Packages
--------

conda
^^^^^

`kessler` is available on [Anaconda](https://anaconda.org/conda-forge/kessler).
To install it via `conda`, we add `conda-forge` channel to the channels:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict

Now, we can install `kessler` either through `conda`:

.. code-block:: console
   
   $ conda install kessler

or `mamba`:

.. code-block:: console
   
   $ mamba install kessler



pip
^^^

`kessler` is available on [Pypi](https://pypi.org/project/kessler/). You can install it via `pip` as:

.. code-block:: console
   
   $ pip install kessler

Installation from source
------------------------


Using ``git``:

.. code-block:: console

   $ git clone https://github.com/kesslerlib/kessler

We follow the usual PR-based development workflow, thus kessler's ``master``
branch is normally kept in a working state.

Verifying the installation
--------------------------

You can verify that kessler was successfully compiled and
installed by running the tests. To do so, you must first install the
optional dependencies.

.. code-block:: bash

   $ pytest

If this command executes without any error, then
your kessler installation is ready for use.

Getting help
------------

If you run into troubles installing `kessler`, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/kesslerlib/kessler/issues>`__.