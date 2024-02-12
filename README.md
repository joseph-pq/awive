# AWIVE
AWIVE, an acronym for Adaptive Water Image Velocimetry Estimator, is a
 software package designed for estimating the velocity field from a sequence of
 images. It comprises two methods: STIV and OTV, both geared towards achieving
 velocity estimations with low computational costs.

## Installing

Install and update using pip:

```bash
pip install awive
```

## Usage

If you want to know how to configure the json file, use the [awive configurator](https://github.com/JosephPenaQuino/awive-configurator)
Execute the commands below:

```
pyenv local 3.11.2
poetry install
poetry run python -m awive.otv river-brenta d0000 -v
```
