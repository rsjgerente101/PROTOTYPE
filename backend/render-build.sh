#!/usr/bin/env bash
set -euo pipefail
# Upgrade packaging tools and install binary-friendly packages on Render
python -m pip install --upgrade pip setuptools wheel
# Install numpy first to ensure a binary wheel is available
python -m pip install --upgrade numpy
# Install remaining requirements, preferring binary wheels when possible
python -m pip install --upgrade --prefer-binary -r backend/requirements.txt
