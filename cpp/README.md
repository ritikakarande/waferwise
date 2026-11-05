# C++ inference

`waferwise_infer` classifies PNG wafer maps with OpenVINO and applies the confidence gate calibrated in Python. Each wafer
is either decided automatically or marked `REVIEW` for a person.

## Build (Windows)

You need:

- Visual Studio 2022 Build Tools with the C++ workload
- the project venv: the OpenVINO wheel includes the C++ SDK, and `requirements.txt` provides CMake and Ninja
- OpenCV: download `opencv-5.0.0-windows.exe` from the [OpenCV releases](https://github.com/opencv/opencv/releases) and
  unpack it into `third_party/` (git-ignored)

```bat
opencv-5.0.0-windows.exe -othird_party -y
```

Then open the "x64 Native Tools Command Prompt for VS 2022" at the repository root:

```bat
.venv\Scripts\activate
cmake -S cpp -B cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DOpenVINO_DIR=%CD%\.venv\Lib\site-packages\openvino\cmake ^
  -DOpenCV_DIR=%CD%\third_party\opencv\build
cmake --build cpp\build
```

## Run

The tool needs the OpenVINO and OpenCV DLLs on `PATH`:

```bat
set PATH=%CD%\.venv\Lib\site-packages\openvino\libs;%CD%\third_party\opencv\build\x64\vc16\bin;%PATH%

cpp\build\waferwise_infer.exe --model outputs\deploy\wafer_cnn_int8.xml --gate outputs\deploy\gate.json ^
  --input outputs\deploy\wafers_val --output outputs\deploy\cpp_val.csv
```

| Option | Meaning |
|---|---|
| `--model` | OpenVINO model (`.xml`) |
| `--gate` | threshold, image size and class names written by `scripts/calibrate_gate.py` |
| `--input` | one PNG, or a folder of PNGs |
| `--output` | results file, default `results.csv` |
| `--batch` | wafers per inference call, default 1 |
| `--device` | OpenVINO device, default `CPU` |

Input PNGs are single-channel wafer maps at any size: black is off wafer, grey is a passing die and white is a failing
die. `scripts/export_wafer_images.py` writes them from the dataset.

The output has one row per wafer, and the console shows a summary with timings:

```
file,predicted,confidence,decision
outputs/deploy/wafers_val\wafer_000000.png,none,0.980990,AUTO
```

## Check against Python

```bash
python scripts/export_wafer_images.py --split val --out outputs/deploy/wafers_val
python scripts/check_cpp.py --csv outputs/deploy/cpp_val.csv --split val
```

Built and tested with MSVC 19.44, OpenVINO 2026.4 and OpenCV 5.0.0. On all 25,955 val wafers, labels and gate decisions
match Python exactly.
