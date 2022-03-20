#!/usr/bin/env bash
pytest -n auto -x -rA --cov=./ --cov-report xml
