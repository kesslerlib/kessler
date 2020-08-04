## Kessler

Kessler is a Python package for simulation-based inference and machine learning for space collision assessment and avoidance.


### How to install

To install kessler, do the following:

```
git clone https://gitlab.com/frontierdevelopmentlab/fdl-europe-2020-constellations/kessler.git
cd kessler
pip install -e .
```

### How to run it in Docker

#### Build the Docker image

In the root folder of this repository, run:
```
docker build -t kessler .
```

#### Run Jupyter inside Docker

```
docker run --rm -it --net=host kessler jupyter notebook --allow-root
```
