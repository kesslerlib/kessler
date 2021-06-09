How to install kessler
=====================

kessler needs Python 3 and PyTorch.

In order to install PyTorch, follow the `official PyTorch instructions <https://pytorch.org/>`__ for your system.

Install via PyPI
----------------

You can install kessler via the `kessler package <https://pypi.org/project/kessler/>`__ listed in the PyPI package index.

.. code-block:: bash

   pip install kessler

Install from source
-------------------

You can install kessler by cloning the latest code and installing locally from the source.

.. code-block:: bash

   git clone https://github.com/kessler/kessler.git
   cd kessler
   pip install .

Docker
------

To build a `Docker <https://www.docker.com/>`__ image for kessler, first `install Docker for your system <https://hub.docker.com/search/?type=edition&offering=community>`__ and then do the following.

.. code-block:: bash

   git clone https://github.com/kessler/kessler.git
   cd kessler
   docker build -t kessler .
