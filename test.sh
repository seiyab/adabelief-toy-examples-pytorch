#!/bin/bash
set -eu
tests=$(find src -name "test_*")
pipenv run python -m unittest -v $tests

