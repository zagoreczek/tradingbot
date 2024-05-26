#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y build-essential
apt-get install -y python3-dev
apt-get install -y libta-lib0-dev
apt-get install -y libta-lib0

# Install TA_Lib
pip install numpy
pip install ta-lib
