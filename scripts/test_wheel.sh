
#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Find the wheel built by maturin.
wheels=(dist/*.whl)

if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one wheel in dist/, found ${#wheels[@]}"
  exit 1
fi

# Create a clean Python environment.
python -m venv .wheel-test

if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
  PYTHON=".wheel-test/Scripts/python.exe"
else
  PYTHON=".wheel-test/bin/python"
fi

# Install the actual wheel, not an editable source build.
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install "${wheels[0]}" pytest onnx

# Verify the installed package and CLI.
"$PYTHON" -I -m eventcv --version
"$PYTHON" -I -m eventcv info data/test/example.npz

# Verify that the wheel includes ONNX support and its bundled runtime.
"$PYTHON" -I -c '
import eventcv

assert "onnx" in eventcv._rust.__features__, \
    "Wheel was built without ONNX support"

runtime = eventcv._ort.find_runtime()
assert runtime is not None, "ONNX Runtime not found"
assert runtime[1] == "bundled", \
    f"Expected bundled ONNX Runtime, got {runtime[1]}"

print("Bundled ONNX Runtime verified:", runtime[0])
'

# Run existing ONNX integration tests against the wheel.
"$PYTHON" -m pytest tests/test_model.py -v -ra

echo "Installed wheel verification passed."
