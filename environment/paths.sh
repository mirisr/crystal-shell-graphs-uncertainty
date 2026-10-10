#!/usr/bin/env bash

export CRYSTAL_UQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$CRYSTAL_UQ_ROOT:$CRYSTAL_UQ_ROOT/notebooks/ispy2${PYTHONPATH:+:$PYTHONPATH}"
