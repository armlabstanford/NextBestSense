#!/bin/bash

PATH="${PATH}:/home/user/.local/bin"
cd ..
cd FisherRF-ns/

git clone https://github.com/JiangWenPL/modified-diff-gaussian-rasterization-w-depth
cd modified-diff-gaussian-rasterization-w-depth/
pip install -e . -v
cd ..
# install nerfstudio
pip install .
