# Hardware and software failure modes when deploying a real-time pose-estimation app on Jetson Orin Nano 8GB, with fixes

Scope: Python 3 Flask rehab monitor, with OpenCV capture (Android IP camera over MJPEG/RTSP, or USB webcam), MediaPipe Pose on the CPU or YOLOv8n-pose (PyTorch or a TensorRT FP16 engine), MJPEG output to a browser, SQLite logging and tegrastats telemetry. Target: Orin Nano 8GB dev kit, JetPack 5.x/6.x, with notes on JetPack 7.2 / Ubuntu 24.04.
Research date: 2026-11-25. Sources are mostly NVIDIA docs, the Ultralytics docs and NVIDIA developer forum threads. Forum claims are labelled as such.

## Q1. Thermal throttling: thresholds, fan behaviour, how it shows up, FPS impact

### Takeaway
Orin Nano thermal throttling starts only at about 99 °C, so in a cooled dev kit the usual cause of clock drops is the power limit (see Q2), not heat. The fan's default "quiet" profile can leave the board hot under a sustained load. Run `sudo jetson_clocks --fan`, or switch to the "cool" profile.

### Cited Findings
- Thermal trip points for the Orin NX/Nano cpu, gpu, cv0–2 and soc0–2 thermal zones: SW throttling at 99.0 °C, HW throttling at 103.0 °C, SW shutdown at 104.5 °C, HW shutdown at 105.0 °C. — [Jetson Linux r36.4 Dev Guide, Platform Power & Performance](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Hardware throttling cuts clocks in three levels: Light 50%, Medium 75%, Heavy 87.5%. — [same Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- The NX/Nano fan profiles are keyed on TMARGIN, the margin below the trip temperature. The default is "quiet": full speed (PWM 255, about 6000 RPM) at a margin of 0–10 °C, about 4000 RPM (PWM 187) at 11–31 °C, and off at 70 °C or more. The "cool" profile runs PWM 255 / 6000 RPM from 0 to 35 °C of margin. — [same Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Since L4T 32.4, `jetson_clocks` no longer sets the fan to maximum by default. `sudo jetson_clocks --fan` forces maximum PWM. — [same Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- tegrastats reports CPU, GPU and EMC frequencies and the temperatures of registered thermal zones. A drop in GPU or CPU frequency (for example `GR3D_FREQ x%@[freq]`) while the temperature is high is how throttling shows up. — [same Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- The on-screen message "System throttled due to over-current" uses the same wording whether the cause is heat or power. A moderator called the message "misleading". In the thread, the board was at about 50–60 °C, well below the thermal trip points, yet it was still throttled by the power limit. — [NVIDIA forum: overcurrent while running TensorRT](https://forums.developer.nvidia.com/t/system-throttled-due-to-overcurrent-while-running-tensorrt/286431)
- Ultralytics recommends running `sudo nvpmodel -m 0` and `sudo jetson_clocks`, and monitoring with jetson-stats (`sudo pip install jetson-stats`, reboot, then `jtop`). — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences
- On the stock dev kit heatsink and fan, the board rarely reaches 99 °C. If FPS drops while the board is at 50–80 °C, suspect the OC (power) throttle first; check it with the `oc*_event_cnt` counters in Q2.
- The worst case is Heavy HW throttling at 87.5%, which cuts clocks to one-eighth. Inference time would grow several-fold, so for example a 10 ms YOLO step could take 40–80 ms. This figure is extrapolated from the clock-cut percentages; it was not measured.
- Mitigation for a box-enclosed deployment: use `jetson_clocks --fan` or the "cool" profile (`/etc/nvfancontrol.conf`, FAN_DEFAULT_PROFILE), keep the vents clear, and log tegrastats temperatures to SQLite next to FPS so the two can be correlated.

### Gaps
- I found no measured FPS-versus-temperature curve for Orin Nano running YOLOv8n-pose or MediaPipe.

## Q2. Power modes, jetson_clocks, power supply, "System throttled due to over-current"

### Takeaway
The Orin Nano 8GB has three power modes: 15W (ID 0, the default), 7W (ID 1) and MAXN / "MAXN SUPER" (ID 2). MAXN is available only with the `jetson-orin-nano-devkit-super` flash configuration from JetPack 6.2 onward. The "over-current" warning usually means the board hit the mode's average (OC2) or instantaneous (OC3) power cap, or an under-voltage dip (OC1) below about 4.5 V. It is not an error. Detect it with the `oc*_event_cnt` counters.

### Cited Findings
- Orin Nano 8GB nvpmodel table:

  | Mode | ID | CPU | CPU max (MHz) | GPU TPC | GPU max (MHz) | EMC max (MHz) |
  |---|---|---|---|---|---|---|
  | 15W | 0 | 6 cores | 1510.4 (1497.6 on `-super`) | 4 | 624.75 (612 on `-super`) | 2133 |
  | 7W | 1 | 4 cores | 960 | 2 | 408 | 2133 |
  | MaxN | 2 | 6 cores | 1728 | 4 | 1020 | 3199 |

  MaxN requires the `jetson-orin-nano-devkit-super` flash config. — [Jetson Linux r36.4 Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- JetPack 6.2 added a 25W mode and the uncapped MAXN SUPER mode. The Orin Nano 8GB supports 15W, 25W and MAXN SUPER. Super Mode on the dev kit needs the `jetson-orin-nano-devkit-super.conf` flash config. — [JetPack 6.2 Super Mode blog / forum](https://forums.developer.nvidia.com/t/jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/320344); [JetPack 6.2 release notes](https://docs.nvidia.com/jetson/jetpack/6.2/release-notes/index.html)
- Mode indices differ by module and config. One forum post gives `sudo nvpmodel -m 2` for MAXN SUPER on Orin Nano, where Orin NX uses `-m 0`. Verify with `nvpmodel -q --verbose`. — [Super Mode forum thread](https://forums.developer.nvidia.com/t/jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/320344)
- Mode switching has known problems. One dev kit was "stuck on 25W" and would not switch. On JetPack 7.2, 25W and MAXN SUPER were missing from the GUI menu and the setting did not persist across reboots, although they could still be set with nvpmodel in a terminal. — [forum: stuck on 25W](https://forums.developer.nvidia.com/t/orin-nano-super-developer-kit-wont-switch-to-maxn-super-or-any-other-mode-stuck-on-25w/350500); [forum: 25W and MAXN SUPER not seen in JetPack 7.2](https://forums.developer.nvidia.com/t/25w-and-maxn-super-not-seen-in-jetpack-7-2/372627)
- `jetson_clocks` pins the CPU, GPU and EMC to the maximum the current nvpmodel mode allows. Options are `--show`, `--store [file]`, `--restore [file]` and `--fan`. **After running jetson_clocks you cannot change the nvpmodel mode without a reboot.** — [Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- INA3221 VDD_IN over-current and under-voltage limits for the Orin Nano 8GB:
  - 15W mode: average power (OC2) 15 W, throttles CPU/GPU 50%. Instantaneous (OC3) 20 W, throttles 87.5%. Under-voltage (OC1) at about 4.5 V, throttles 87.5%.
  - 25W mode (`-super`): OC2 25 W and OC3 25 W, both 50%. OC1 at about 4.5 V, 50%.

  — [Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Detection commands: `grep "" /sys/class/hwmon/hwmon<x>/oc*_throt_en` shows which events are enabled, and `grep "" /sys/class/hwmon/hwmon<x>/oc*_event_cnt` shows how many have fired. The rail readings `in1_input` (mV), `curr1_input` (mA), `curr1_crit` and `curr1_max` are under `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon<x>/`. — [Dev Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- NVIDIA moderators say that throttling lowers clocks to keep the board under the mode's power budget. The fix is to choose or customise an nvpmodel mode, not to treat it as a fault. Use VDD_IN for total module power; do not add up the other rails, which was a mistake made in the thread. — [forum 286431](https://forums.developer.nvidia.com/t/system-throttled-due-to-overcurrent-while-running-tensorrt/286431)
- Forum users report that the dev kit carrier's DC jack (DCJ_IN) is rated for 4.2 A maximum. Others report the warning even with the official adapter. One reply says the warning appears when the input drops below 4.5 V, which matches OC1 in the Dev Guide. — [SparkFun / NVIDIA forum threads](https://community.sparkfun.com/t/jetson-orin-nano-overcurrent/64960); [forum: Orin NX/Nano over-current on screen](https://forums.developer.nvidia.com/t/jetson-orin-nx-nano-shows-system-throttled-due-to-over-current-on-screen/368879)

### Inferences
- For a rehab app at 30 FPS, YOLOv8n-pose TRT FP16 should fit within the 15W mode. MAXN SUPER mainly buys GPU headroom: 1020 MHz against 625 MHz for the GPU, and 3199 MHz against 2133 MHz for memory. That headroom matters if an LLM or a PyTorch model runs at the same time.
- Use the official 19 V barrel-jack supply. The dev kit is powered through its DC jack; I found no source documenting USB-C power input on this dev kit, so I am not claiming it. Under-voltage (OC1) points to the supply or cable. Repeated OC2/OC3 events point to workload against the mode budget.
- Before running `jetson_clocks`, run `jetson_clocks --store` so the settings can be restored later. Set the nvpmodel mode first, because the mode cannot be changed afterwards without a reboot.

### Gaps
- The visible Dev Guide text did not say which of oc1/oc2/oc3 maps to which event type (the docs list them as Under Voltage, Average, Instantaneous).
- I found no official per-mode power figure for MAXN SUPER, which is uncapped.
- I did not verify the exact dev kit supply rating (19 V, about 2.37 A / 45 W adapter); it is not sourced here.

## Q3. 8 GB unified memory: OOM killer, swap/zram, pose model plus an LLM, TRT vs PyTorch memory

### Takeaway
The CPU and GPU share the same 8 GB, so the desktop, PyTorch's CUDA context and an LLM all draw from one pool. To keep headroom: boot headless (saves about 800–865 MB), use a TensorRT engine instead of PyTorch, disable zram and put swap on NVMe, and keep any LLM below about 4B parameters.

### Cited Findings
- The Jetson AI Lab says disabling the desktop frees about 800 MB. Commands: `sudo init 3` for a temporary switch, or `sudo systemctl set-default multi-user.target` to make it permanent (`graphical.target` restores it). — [Jetson AI Lab RAM optimization](https://www.jetson-ai-lab.com/tutorials/ram-optimization/)
- Ultralytics estimates the savings as about 865 MB from a headless boot, about 32 MB from disabling unused services, and about 200 MB or more from running inference with `show=False`, that is, without a display window. — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- The Jetson AI Lab's swap recipe is: `sudo systemctl disable nvzramconfig`, then `sudo fallocate -l 16G /ssd/16GB.swap`, `sudo mkswap ...`, `sudo swapon ...`, and an fstab line `/ssd/16GB.swap none swap sw 0 0`. It recommends NVMe over SD for the swap file. Optionally run `sudo systemctl disable nvargus-daemon.service` if no CSI camera is used. — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/ram-optimization/)
- Forum reports:
  - On an 8 GB Orin Nano, zram alone ran out of memory and on-disk swap was needed.
  - A VILA 1.5 3B run with 20 GB of swap locked the device up.
  - An NVIDIA reply recommends models under 4B parameters on Orin Nano.
  - Ollama services have been OOM-killed.

  — [forum: zram on Orin](https://forums.developer.nvidia.com/t/zram-swap-on-jetson-orin-efficient-enough/277490); [VILA 1.5 3B](https://forums.developer.nvidia.com/t/vila-1-5-3b-on-jetson-orin-nano/293000); [Ollama oom-kill](https://forums.developer.nvidia.com/t/ollama-service-failed-with-result-oom-kill/374647); [YOLO + LLM thread](https://forums.developer.nvidia.com/t/jetson-orin-nano-deploying-yolo-with-llm/359823)
- There are reports of TensorRT OOM on JetPack 6.2, and of the nanoLLM subprocess being SIGKILLed on JetPack 6.1. — [JetPack 6.2 + TensorRT OOM](https://forums.developer.nvidia.com/t/jetpack6-2-tensorrt-oom-issue/322602); [nanoLLM SIGKILL](https://forums.developer.nvidia.com/t/error-running-nanollm-on-jetson-orin-nano-with-jetpack-6-1-subprocess-sigkill-issue/318775)

### Inferences
- Symptoms to watch for: the process dies with "Killed" or exit code 137, `dmesg | grep -i oom` shows OOM entries, or `journalctl` shows `oom-kill`. GPU allocations can also fail with CUDA out-of-memory even though free RAM appears available.
- GPU memory comes from the same shared pool and swap cannot back it, so swap helps CPU-side pages but cannot rescue GPU allocations. This follows from the unified-memory design; I did not find an NVIDIA document stating it.
- Prefer one process holding the model. Flask debug mode with the reloader, or multiple gunicorn workers, would load the model two or more times.

### Gaps
- I found no sourced measurement of resident memory for YOLOv8n-pose under PyTorch versus TensorRT on Orin Nano. From general experience, importing torch plus the CUDA context typically costs well over 1 GB, while a TensorRT runtime with a roughly 10 MB nano engine is much smaller. Treat this as unverified and measure it with `tegrastats` RAM or jtop.

## Q4. MediaPipe on Jetson ARM64

### Takeaway
`pip install mediapipe` now installs on aarch64, but only as a CPU build. The TFLite GPU delegate does not run on Jetson's CUDA GPU through the PyPI wheel. People who wanted GPU acceleration built from source, and older community wheels target JetPack 4.x only. On the Orin Nano CPU, MediaPipe hand tracking has been reported below 5 FPS.

### Cited Findings
- PyPI lists mediapipe 1.2.3, released 11 Nov 2026, with the wheel `mediapipe-1.2.3-py3-none-manylinux_2_28_aarch64.whl` (40.7 MB). It needs glibc 2.28 or newer, the classifiers list Python 3.9–3.14, and there is no sdist. (I could not cross-check this version number; the wheel filename is from the PyPI page.) — [PyPI mediapipe](https://pypi.org/project/mediapipe/#files)
- In a June 2024 thread, a user who built MediaPipe from source on an Orin Nano got hand tracking under 5 FPS. The log showed TFLite "initializing with the CPU … as the delegate and not the GPU". NVIDIA's reply was to use a CUDA-built package (an OpenCV 4.9 CUDA build script for JetPack 6, sm_87) and pointed to full TensorFlow for Jetson. No MediaPipe GPU fix was given. — [forum: MediaPipe/TFLite GPU delegate on Orin Nano](https://forums.developer.nvidia.com/t/mediapipe-tensorflow-lite-gpu-delegate-on-jetson-orin-nano/296851)
- An NVIDIA reply says "some users have successfully installed mediapipe on the Jetson", but the linked example is for AGX Xavier on JetPack 4.6.1 in Docker. — [forum: Does Orin Nano support MediaPipe](https://forums.developer.nvidia.com/t/does-jetson-orin-nano-support-mediapipe/290797)
- One user reported pip rejecting a self-built MediaPipe wheel on JetPack 6.2 because of its filename tag. Community wheels (0.8.x for JetPack 4.6 / CUDA 10.2) do not match JetPack 6. A Raspberry Pi report shows "not a supported wheel on this platform" when the tag does not match. — [forum: MediaPipe build JP6.2](https://forums.developer.nvidia.com/t/mediapipe-build-for-jetson-orin-nano-jp6-2/329399); [anion0278/mediapipe](https://github.com/anion0278/mediapipe); [mediapipe issue 4673](https://github.com/google-ai-edge/mediapipe/issues/4673)

### Inferences
- Keep MediaPipe as the CPU fallback and YOLOv8n-pose TensorRT as the GPU path on Jetson. Set expectations for MediaPipe Pose on the CPU: well below desktop FPS, depending on model complexity (lite/full/heavy). Use `model_complexity=0` (lite) on Jetson.
- The glibc check is safe on all JetPack versions here: Ubuntu 20.04 (glibc 2.31), 22.04 (2.35) and 24.04 all exceed 2.28. This comes from general knowledge, not a cited source.
- MediaPipe wheels may pull in `numpy` and `opencv-contrib-python` and replace the JetPack OpenCV (see Q7). Check `pip show` after installing.

### Gaps
- I found no sourced FPS figure for MediaPipe Pose specifically, as opposed to Hands, on the Orin Nano CPU.

## Q5. TensorRT .engine files: portability, export, FP16/INT8 issues

### Takeaway
A `.engine` file is tied to the TensorRT version and GPU compute capability it was built with (Orin is sm_87), and by default to the device type. Copying it between machines or JetPack versions fails with "Serialization assertion … Version tag does not match" or "magicTag" errors. Always build the engine on the target Jetson, ship the `.pt` or `.onnx`, and rebuild on first run.

### Cited Findings
- An engine is compatible only with the device type it was built on by default. The plan records the TensorRT version and the compute capability and checks both at load time. Engines built with a newer TensorRT major version generally will not load on an older runtime. — [TensorRT Engine Compatibility docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- These errors were seen on Jetson after moving engines between devices or TensorRT versions: `Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match`, and `Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed`. Regenerating the engine on the device fixed them. A JetPack 6 (L4T 36.4.0) report shows the same mismatch. — [forum 309261](https://forums.developer.nvidia.com/t/serialization-assertion-stdversionread-kserialization-version-failed-version-tag-does-not-match/309261); [forum 300099](https://forums.developer.nvidia.com/t/error-code-1-serialization-serialization-assertion-plan-header-magictag-rt-kplan-magic-tag-failed/300099); [forum 353515](https://forums.developer.nvidia.com/t/erroneous-mismatch-in-tensorrt-serializations-on-jetpack-6-l4t-36-4-0/353515)
- Ultralytics notes:
  - Export with `yolo export model=<m>.pt format=engine` (add `half=True` for FP16).
  - For pose, a dynamic-shape export failed with a Myelin "Could not find any implementation" error; use `imgsz=640 batch=1 dynamic=False`.
  - INT8 calibration should run on the target device.
  - TensorRT 10.3.0 (JetPack 6) cannot build INT8 engines; upgrade to 10.7.0 or later.
  - TensorRT 10.7 is the last release with DLA support.
  - TensorRT 11 is unsupported on JetPack.

  — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Ultralytics Orin Nano Super benchmark for YOLO26n detection (not pose), inference only, in ms per image: PyTorch 15.60, TRT FP32 7.53, TRT FP16 4.57, TRT INT8 3.80. mAP50-95: 0.4790 (PyTorch), 0.4800 (FP16), 0.4490 (INT8), so INT8 loses about 3 points. — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- `pip install tensorrt` fails on Jetson with "TensorRT does not currently build wheels for Tegra systems". Use the JetPack-provided TensorRT. — [Ultralytics guide, community note](https://docs.ultralytics.com/guides/nvidia-jetson/)
- One forum report: "TensorRT 10.7.0 on Orin Nano: Target GPU SM 87 is not supported by this TensorRT release". This shows that mixing a non-JetPack TensorRT build in can fail. — [forum 364236](https://forums.developer.nvidia.com/t/tensorrt-10-7-0-on-orin-nano-target-gpu-sm-87-is-not-supported-by-this-tensorrt-release/364236)

### Inferences
- Pose models behave like detection models on TRT: expect FP16 to be about 3x faster than PyTorch at roughly equal accuracy. Skip INT8 for rehab joint-angle accuracy unless it is validated against FP16.
- In the app, use the engine only if it exists and loads. Wrap deserialization in try/except and fall back to re-export (minutes on device) or to PyTorch. Store the TensorRT version next to the engine, for example in the filename `yolov8n-pose_trt10.3_sm87_fp16.engine`, and rebuild when it differs.

### Gaps
- I found no sourced export time for YOLOv8n-pose to TRT on Orin Nano. Anecdotally it takes several minutes; measure it.
- The Ultralytics benchmark page covers YOLO26 detection. I found no official YOLOv8n-pose Orin Nano numbers.

## Q6. PyTorch/torchvision on Jetson: wheels, CUDA not available, NumPy 2.x

### Takeaway
PyTorch from PyPI or pytorch.org for aarch64 is a CPU-only build, so `torch.cuda.is_available()` returns False. Use NVIDIA/Jetson-specific wheels matched to the JetPack and Python version: cp38 for JetPack 5, cp310 for JetPack 6, and cp312 with CUDA 13 for JetPack 7. NVIDIA's JetPack 6 wheels are built against NumPy 1.x, which clashes with NumPy 2.x.

### Cited Findings
- Ultralytics' tested wheels:
  - **JetPack 5.1.2 (Python 3.8):** `torch-2.1.0a0+41361538.nv23.06-cp38`, `torchvision-0.16.2+c6f3977-cp38`, and onnxruntime-gpu 1.17.0. Then run `pip install numpy==1.23.5`, because onnxruntime-gpu upgrades NumPy.
  - **JetPack 6.1 (Python 3.10):** `torch-2.10.0-cp310`, `torchvision-0.25.0-cp310` (install cuDSS 0.7.1 first) and onnxruntime-gpu 1.23.0 cp310, all from `github.com/ultralytics/assets/releases/download/v0.0.0/`.
  - **JetPack 7.2:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`, plus onnxruntime-gpu 1.24.0 cp312.

  — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- A community report: torch 2.10.0 on JetPack 6.1 raised `ImportError libcublas.so.13`, and the user fell back to torch 2.8.0. Check the L4T version with `cat /etc/nv_tegra_release`. — [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Wheels from pytorch.org for generic ARM64 were "not built with CUDA enabled". — [forum: PyTorch 6.1 GPU not working](https://forums.developer.nvidia.com/t/in-pytorch-6-1-the-gpu-is-not-working/314143)
- NumPy ABI clash:
  - The jp6/cu126 Torch wheel is built with NumPy 1.x and fails with the "compiled using NumPy 1.x cannot be run in NumPy 2" error.
  - Downgrading NumPy below 2 can break an OpenCV CUDA build compiled against NumPy 2.
  - A user who built PyTorch 2.8 from source with NumPy 2 got `torch.version.cuda == None`, a CPU build.

  — [forum 370484](https://forums.developer.nvidia.com/t/pytorch-from-jp6-cu126-on-jetson-orin-nano-jetpack-6-2-1-cuda-12-6-failing-due-to-numpy-mismatch/370484); [forum 350909](https://forums.developer.nvidia.com/t/pytorch-gpu-build-on-jetson-agx-orin-jetpack-6-1-cuda-12-6-failing-due-to-numpy-abi-mismatch/350909); [forum 368698](https://forums.developer.nvidia.com/t/build-pytorch-2-8-with-cuda-numpy-2-on-jetpack-6-2/368698); [forum: numpy~=1.26 conflicts](https://forums.developer.nvidia.com/t/numpy-package-conflicts-forcing-to-install-numpy-1-26/328035)
- Another thread with `torch.cuda.is_available()` False on JetPack 6.0 DP (Orin NX). — [forum 277189](https://forums.developer.nvidia.com/t/jetson-orin-nx-pytorch-2-1-2-on-jetpack-6-0-dp-torch-cuda-is-available-is-false/277189/9)

### Inferences
- Diagnostic one-liner: `python3 -c "import torch,numpy;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),numpy.__version__)"`. If `torch.version.cuda` is None, the wheel is wrong. If it shows a CUDA version but import fails on NumPy, pin `numpy<2` (for example 1.26.x on JetPack 6, or 1.23.5 on JetPack 5).
- Pin `numpy<2` in the Jetson requirements file. `pip install ultralytics` or `mediapipe` must not pull a CPU torch from PyPI over the NVIDIA one: install torch first, then the others, and check `torch.cuda.is_available()` afterwards.

### Gaps
- The exact NVIDIA-hosted index URL for JetPack 6.2 (`pypi.jetson-ai-lab...`/jp6/cu126) was mentioned in threads but not fetched or verified here.

## Q7. OpenCV: pip vs JetPack build, GStreamer, low-latency RTSP

### Takeaway
`pip install opencv-python` installs a build without GStreamer or CUDA, and it shadows JetPack's apt OpenCV, which does have GStreamer. Uninstall it, use `python3-opencv` from apt (or a CUDA build), and use a GStreamer pipeline ending in `appsink drop=true max-buffers=1 sync=false` to keep RTSP latency low.

### Cited Findings
- The PyPI `opencv-python` wheel lacks GStreamer and can shadow JetPack's OpenCV. The fix is to uninstall it and run `sudo apt install -y python3-opencv`. — [Ultralytics Jetson guide (community report)](https://docs.ultralytics.com/guides/nvidia-jetson/)
- NVIDIA staff supplied an OpenCV 4.9.0 build script for JetPack 6.0, giving CUDA 12.2, arch 87 and cuDNN 8.9.4 under Python 3.10.12, for CUDA-enabled OpenCV. — [forum 296851](https://forums.developer.nvidia.com/t/mediapipe-tensorflow-lite-gpu-delegate-on-jetson-orin-nano/296851)
- Pipeline pattern from the Jetson forums: `rtspsrc location=... latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false`. `appsink max-buffers=1 drop=True` lowers latency at the cost of dropped frames. Users also report `drop-on-latency=true` on rtspsrc, and leaky queues for 5–6 s delay problems. — [forum: Increasing latency with GStreamer in OpenCV](https://forums.developer.nvidia.com/t/increasing-latency-when-using-gstreamer-in-opencv/282046); [TX2 latency thread](https://forums.developer.nvidia.com/t/how-to-decrease-the-latency-in-tx2-when-decoding-a-rtsp-stream/75808); [JetPack 6.2 RTSP NV decoder](https://forums.developer.nvidia.com/t/jetpack-6-2-rtsp-stream-enable-nvidia-decoder/324240); [Nano RTSP increasing delay](https://forums.developer.nvidia.com/t/jetson-nano-rtsp-stream-constant-increase-in-delay/169030/8)
- `CAP_PROP_BUFFERSIZE` is documented on Jetson forums only for the V4L backend, where it is used to hold one frame. It is not shown to work for the GStreamer or FFmpeg backends. — [forum: Delay with GStreamer in OpenCV 3.4.2](https://forums.developer.nvidia.com/t/delay-with-gstreamer-in-opencv-3-4-2/67784)
- One report on Orin NX: nvv4l2decoder could not feed appsink directly, so an `nvvidconv` to system-memory BGRx step is needed. — [forum 367293](https://forums.developer.nvidia.com/t/nvv412-cant-use-with-appsink-on-orin-nx/367293)

### Inferences
- Check the build with `python3 -c "import cv2;print(cv2.getBuildInformation())" | grep -E "GStreamer|CUDA"`. If GStreamer shows NO, a pip wheel is shadowing the apt build. Run `pip list | grep opencv` and remove every `opencv-*` pip package, including `opencv-contrib-python` pulled in by MediaPipe.
- For HTTP MJPEG streams such as the Android "IP Webcam" app's `/video`, nvv4l2decoder does not apply because it is for H.264/H.265 (nvjpegdec exists for JPEG but was not researched here). The most robust low-latency pattern is a dedicated capture thread that continuously calls `cap.read()` and keeps only the latest frame, so stale buffered frames never reach inference.

### Gaps
- I found no authoritative glass-to-glass latency numbers for OpenCV+GStreamer RTSP on Orin Nano.

## Q8. IP camera over Wi-Fi: MJPEG vs RTSP, buffering, drops, reconnects

### Takeaway
I found no authoritative measurements. Latency accumulates in the decoder or OpenCV buffers when inference runs slower than the camera frame rate. The fix is architectural: a latest-frame grabber thread, appsink drop, and reconnect with backoff.

### Cited Findings
- Constantly increasing delay on RTSP streams is a known Jetson symptom when frames are consumed more slowly than they arrive. The fixes are leaky queues and appsink drop. — [forum: Nano RTSP constant increase in delay](https://forums.developer.nvidia.com/t/jetson-nano-rtsp-stream-constant-increase-in-delay/169030/8); [forum 282046](https://forums.developer.nvidia.com/t/increasing-latency-when-using-gstreamer-in-opencv/282046)
- RTSP input jitter, meaning frames not arriving every 33.33 ms, has been reported on Jetson. — [forum: RTSP input jitter](https://forums.developer.nvidia.com/t/rtsp-input-jitter-not-stable-33-33ms-frames/228657)

### Inferences (not sourced, engineering judgement)
- Reconnect logic: if `cap.read()` returns False or no frame arrives for more than about 2 s, release and reopen with exponential backoff, and show a "camera offline" frame in the MJPEG output so the Flask stream does not hang.
- For MJPEG over HTTP, each frame is a full JPEG, so bandwidth is high but there is no GOP or keyframe delay. RTSP/H.264 uses less bandwidth but adds a jitter buffer; rtspsrc's default latency is 2000 ms unless set lower (GStreamer default, from memory). Over a congested 2.4 GHz link, MJPEG at a reduced resolution (640x480) is often the simplest choice. Prefer 5 GHz, or a phone hotspot or USB tethering.

### Gaps
- I found no sourced typical latency figures (ms) for Android IP Webcam MJPEG or RTSP over Wi-Fi, or for USB UVC on Jetson.

## Q9. Storage: microSD corruption, SD vs NVMe, SQLite durability

### Takeaway
SD-card filesystem corruption on the Orin Nano is widely reported. Forum replies call it a known issue and say it is improved in JetPack 5.1.4 / 6.1. Any writable filesystem can corrupt on sudden power loss. Boot from NVMe for deployment, shut down cleanly, and configure SQLite for crash safety.

### Cited Findings
- A forum reply calls Orin Nano SD card corruption a known issue, suggests booting from NVMe or USB, and says the issue was resolved in JetPack 5.1.4 / 6.1. I did not verify this against the release notes. A moderator noted that in such cases the bootloader works but the rootfs cannot be mounted. — [forum 315022](https://forums.developer.nvidia.com/t/frequent-sd-card-issues-and-boot-failures-on-jetson-orin-nano-8gb-developer-kit/315022); [forum 272404](https://forums.developer.nvidia.com/t/jetson-orin-nano-booting-issue-sd-card-corruption-problem/272404)
- "Any filesystem which is mounted with write ability can have the filesystem damaged by sudden power loss," and that includes NVMe. One JetPack 7.2 user reports repeated ext4 corruption on NVMe with "Attempting recovery boot". — [forum 260041](https://forums.developer.nvidia.com/t/the-integrity-of-the-jetson-file-storage-system/260041); [forum 379702](https://forums.developer.nvidia.com/t/jetson-orin-nano-repeated-ext4-nvme-filesystem-corruption-and-attempting-recovery-boot-on-jetpack-7-2/379702)
- A read-only rootfs is discussed for deployments. — [forum: RO SD card filesystem](https://forums.developer.nvidia.com/t/ro-sd-card-filesystem-for-deploying/259832)

### Inferences
- SQLite settings: `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL` (or FULL for maximum durability). Batch inserts once per rep or per second rather than once per frame. Use one writer connection, because Flask threads plus SQLite risk "database is locked" errors, and set `timeout=` on connect. This is standard SQLite practice and was not sourced in this session.
- Put the DB, logs and swap on NVMe rather than the microSD card.

### Gaps
- I found no sourced SD-versus-NVMe throughput numbers for Orin Nano.

## Q10. Python/JetPack conflicts: Python versions, PEP 668, venv

### Takeaway
JetPack 5 is Ubuntu 20.04 with Python 3.8, JetPack 6 is Ubuntu 22.04 with Python 3.10, and JetPack 7 is Ubuntu 24.04 with Python 3.12, which is the release that enforces PEP 668. The wheel tags in the Ultralytics guide (cp38, cp310, cp312) confirm the Python versions. TensorRT and the apt OpenCV are system packages, so a venv needs `--system-site-packages` (with the same Python version) to see them.

### Cited Findings
- Wheel tags: JetPack 5.1.2 uses cp38 (Python 3.8), JetPack 6.1 uses cp310 (Python 3.10), and JetPack 7.2 uses cp312 (Python 3.12). — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Fix for "No module named 'tensorrt'" inside a venv: install `sudo apt-get install python3-libnvinfer python3-libnvinfer-dev`, delete the venv, and recreate it with `python3 -m venv .venv --system-site-packages`. Toggling `include-system-site-packages` in an existing venv did not work for one user. The alternative is to symlink the system `tensorrt` package into the venv. — [forum 223360](https://forums.developer.nvidia.com/t/cant-access-tensorrt-from-a-python-venv/223360/5); [forum 315156](https://forums.developer.nvidia.com/t/jetson-orin-nx-modulenotfounderror-no-module-named-tensorrt/315156); [forum 287095](https://forums.developer.nvidia.com/t/cant-import-tensorrt-in-python-with-jetpack-6-0dp-no-module-named-tensorrt/287095)
- TensorRT Python bindings were unavailable inside a conda environment, and installing outside conda fixed it. — [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/); [forum 276829](https://forums.developer.nvidia.com/t/installing-tensorrt-in-a-conda-env-on-jetson-orin-nano/276829)

### Inferences
- PEP 668's "externally-managed-environment" error appears on Ubuntu 23.04 and later, so on JetPack 7.x (24.04) but not on JetPack 5/6. This is from general knowledge. Fix it with a venv using `--system-site-packages`, not `--break-system-packages`.
- Avoid conda on Jetson for this app.

### Gaps
- I did not fetch an official NVIDIA statement of the Python version per JetPack release; the mapping is inferred from wheel tags.

## Q11. Other gotchas

### Takeaway
Run headless, give the service user camera permissions, use `sudo` for full tegrastats power fields, and avoid running the Flask dev server with the reloader.

### Cited Findings
- Headless: `sudo systemctl set-default multi-user.target`, and use `show=False` / no `cv2.imshow`. — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/ram-optimization/); [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- jtop/jetson-stats needs `sudo pip install jetson-stats` and a reboot. — [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Run tegrastats as root and log to a file, for example `sudo tegrastats > tegrastats.txt` as shown in OC-debugging threads. — [forum: over-current threads](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-system-throttled-due-to-over-current-lowcurrent-problem/368504)
- JetPack 7.2 for Orin Nano has no downloadable SD-card image; it installs from ISO onto microSD or NVMe. — [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences (unsourced, general Linux)
- `/dev/video0` "Permission denied": add the user to the `video` group with `sudo usermod -aG video $USER` and log in again. USB webcams may change index after replug; use `/dev/v4l/by-id/...`.
- Calling `tegrastats` from Flask: spawn it once with `--interval 1000` and parse the stream, rather than starting a process per request. Some rail and power fields need root, so run the service via systemd or a sudoers rule for tegrastats only.
- Flask's `debug=True` reloader loads the model twice and doubles memory. Use `use_reloader=False` or a production server with a single worker and threads.

### Gaps
- I did not find an official list of which tegrastats fields require root on JetPack 6.
