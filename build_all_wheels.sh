#!/bin/bash
# build_all_wheels.sh

TORCH_VERSIONS=("2.6.0" "2.7.0")
PYTHON_VERSIONS=("3.11" "3.12")

for torch_ver in "${TORCH_VERSIONS[@]}"; do
    for python_ver in "${PYTHON_VERSIONS[@]}"; do
        echo "Building wheel for torch=$torch_ver python=$python_ver"
        ./build_wheel.sh -t "$torch_ver" -p "$python_ver" -c
    done
done