<div align="center">

# AWIVE

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://docs.python.org/3.9/)
[![opencv](https://img.shields.io/badge/OpenCV_4.6-ee4c2c?logo=opencv&logoColor=white)](https://opencv.org/releases/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<br>
[![Code Quality](https://github.com/natek-systems/n-flow/actions/workflows/code-quality.yml/badge.svg)](https://github.com/natek-systems/n-flow/actions/workflows/code-quality.yml)
[![Tests](https://github.com/natek-systems/n-flow/actions/workflows/tests.yaml/badge.svg)](https://github.com/natek-systems/n-flow/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/natek-systems/n-flow/graph/badge.svg?token=JYOVV0D79Iq)](https://codecov.io/gh/natek-systems/n-flow)
<br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template?tab=MIT-1-ov-file)

</div>

<br>

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
