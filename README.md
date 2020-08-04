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

If you're using Linux:
```
docker run --rm -it -v $PWD:/workspace --net=host kessler jupyter notebook --allow-root
```

If you're using MacOS:
```
docker run --rm -it -v $PWD:/workspace -p 8888:8888 kessler jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
