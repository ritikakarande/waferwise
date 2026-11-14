#!/bin/bash
# The steps of .github/workflows/tests.yml for Ubuntu 24.04, run as root from the repository root.
# Called by scripts/rehearse_ci.sh inside a container. Keep it in step with the workflow file.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== 1/6 System libraries for the Python tests"
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 python3 python3-venv ca-certificates > /dev/null

echo "=== 2/6 Python packages"
python3 -m venv /venv
. /venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt -e .

echo "=== 3/6 Tests and demo"
pytest -q
python scripts/demo.py

echo "=== 4/6 Compiler and OpenCV"
apt-get install -y -qq --no-install-recommends build-essential libopencv-dev > /dev/null

echo "=== 5/6 Build the C++ CLI"
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DOpenVINO_DIR="$(python -c 'from openvino.utils import get_cmake_path; print(get_cmake_path())')"
cmake --build cpp/build

echo "=== 6/6 C++ gives the same answers as Python"
WAFERWISE_INFER=cpp/build/waferwise_infer pytest -q tests/test_demo.py

echo "=== All steps passed"
