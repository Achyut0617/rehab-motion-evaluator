# Comparative Edge Hardware Benchmarks for Real-Time Human Pose Estimation (Rehab App: MediaPipe Pose / YOLOv8n-pose, 640x480)

Research date: 2026-11-25. Each number below states its model, input size, precision and software where the source gives them. **Most numbers were measured under different conditions and cannot be compared directly.** The main differences:
- Ultralytics doc tables give **inference-only** time, with pre- and post-processing excluded, on coco128 at imgsz=640. The current Ultralytics pages now list **YOLO26** only and no longer show YOLOv8n or YOLO11n rows.
- Hailo figures come from `hailortcli` raw-chip throughput at **batch 8**. Application FPS at batch 1 is roughly 35–60% lower.
- Coral figures time one inference on a USB Accelerator attached to a desktop host.
- INR values use an **assumed rate of about ₹85/USD** (my assumption, not taken from a source). They exclude Indian import duty and GST, which in practice add a large amount.

---

## Summary comparison table

| Platform | AI compute (INT8 TOPS) | CPU / GPU / accel | RAM | Power envelope | Price (USD → ~INR @85) | Measured YOLO-class FPS / latency (conditions) | Pose-specific measured |
|---|---|---|---|---|---|---|---|
| **Jetson Orin Nano 8GB – original** (JetPack 5/6.0/6.1) | 40 sparse / 20 dense | 6x A78AE @1.5 GHz; 1024 CUDA / 32 Tensor cores @625 MHz; 10 TFLOPS FP16 | 8 GB, 68 GB/s | 7 W / 15 W modes | $499 dev kit (old price) → ~₹42,400 | ~60–100 FPS YOLOv8n TRT FP16 (community; see notes) | none found directly |
| **Jetson Orin Nano 8GB – "Super"** (JetPack 6.2) | 67 sparse / 33 dense | 6x A78AE @1.7 GHz; GPU @1020 MHz; 17 TFLOPS FP16 | 8 GB, 102 GB/s | 15 W / 25 W / MAXN SUPER | $249 dev kit → ~₹21,200 (before duty) | YOLOv8n 640: PyTorch FP32 20.9 ms (48 FPS); TRT FP32 8.4 ms (120 FPS); **TRT FP16 4.43 ms (225 FPS)**; TRT INT8 3.49 ms (286 FPS) (GPU compute only, JP 6.2.2). YOLO26n 640: PyTorch 15.6 ms, ONNX 15.8 ms, TRT FP16 4.57 ms, TRT INT8 3.80 ms (Ultralytics) | none found directly for YOLOv8n-pose (gap) |
| **Jetson Orin Nano 4GB** (orig → Super) | 20/10 → 34/17 | 6x A78AE; 512 CUDA / 16 Tensor @625→1020 MHz; 5→8.5 TFLOPS FP16 | 4 GB, 34→51 GB/s | 7/10 W → 10/25 W/MAXN SUPER | price not found | no direct YOLOv8 numbers found | — |
| **Jetson Orin NX 16GB** (orig → Super) | 100/50 → 157/78 (+DLA 40→80 sparse) | 8x A78AE @2.0 GHz; 1024 CUDA @918→1173 MHz | 16 GB, 102 GB/s | 10–25 W → up to 40 W / MAXN SUPER | not found | YOLO26n 640: PyTorch 13.9 ms; TRT FP16 4.13 ms; TRT INT8 3.49 ms (Ultralytics, JP6.0/5.1.3). YOLO11 TRT FP16 5.27 ms (DeepStream guide) | — |
| **Jetson Orin NX 8GB** (orig → Super) | 70/35 → 117/58 | 6x A78AE @2.0 GHz; 1024 CUDA @765→1173 MHz | 8 GB, 102 GB/s | 10–20 W → up to 40 W | not found | — | — |
| **Jetson Nano 4GB (Maxwell, 2019)** | ~0.5 TFLOPS FP16 (unverified, see gaps) | 4x A57; 128 Maxwell CUDA cores | 4 GB | 5/10 W (unverified) | not found | YOLOv8n ~19 FPS vs ~100 FPS on Orin Nano (Qengineering, TensorRT C++) | — |
| **Raspberry Pi 5 (CPU only)** | n/a | 4x Cortex-A76 @2.4 GHz | 4/8 GB | not found | not found in sources | YOLO26n 640 FP32: PyTorch 299 ms (3.3 FPS); ONNX 126 ms; OpenVINO 105 ms; **NCNN 67 ms (~15 FPS)**. YOLO11n ONNX 147 ms | MoveNet Lightning ~12 ms, Thunder ~30 ms (secondary source) |
| **Pi 5 + AI Kit / AI HAT+ 13 TOPS (Hailo-8L)** | 13 | Hailo-8L over PCIe Gen 3 x1 | host RAM | not found | $70 HAT+ 13T → ~₹6,000 (+Pi) | yolov8s 640: 127.9 FPS (batch 8, raw) | **yolov8s_pose 640: 123.4 FPS (batch 8, raw)**. At batch 1: ~65 FPS (Seeed, via forum) and 46 FPS (user-measured) |
| **Pi 5 + AI HAT+ 26 TOPS (Hailo-8)** | 26 | Hailo-8, PCIe Gen 3 x1 | host RAM | not found | $110 launch ($132 at SparkFun) → ~₹9,400–11,200 | users report results below the official numbers (forum) | — |
| **Google Coral USB / Dev Board (Edge TPU)** | 4 (2 TOPS/W) | Edge TPU; Dev Board has 4x A53 | Dev Board 1 GB / 4 GB (unverified) | ~2 W for the TPU (0.5 W/TOPS) | ~$60 for USB (Geerling) | MobileNet v1 SSD 224: 6.5 ms USB3 / 11 ms Dev Board | **MoveNet Lightning 192x192: 7.1 ms; Thunder 256x256: 13.8 ms; PoseNet MobileNet 481x641: 10.3 ms** (USB Accel + desktop host) |
| **Laptop CPU (Intel i5/i7)** | — | x86 | — | ~15–45 W (unverified) | — | YOLO26n-pose 640 ONNX CPU: 40.3 ms (~25 FPS; Ultralytics, CPU model unspecified) | MediaPipe Pose CPU: gap |
| **Laptop GPU (RTX 3050/4060 class)** | — | — | — | — | — | no source found (gap) | — |
| **Cloud NVIDIA T4** | 130 INT8 (unverified) | Turing, 16 GB | 16 GB | 70 W (unverified) | cloud hourly | **YOLO26n-pose 640 TRT10: 1.8 ms** (Ultralytics) | — |
| **Cloud A10G** | — | — | — | — | — | no source found (gap) | — |

---

## Q1. What FPS does YOLOv8n-pose (or YOLOv8n / YOLO11n / YOLO26n) achieve on each platform, in PyTorch vs TensorRT / ONNX / Hailo / EdgeTPU?

### Takeaway
On Orin Nano Super, a nano-sized YOLO at 640 runs in about **4.4–4.6 ms with TensorRT FP16 (around 220 FPS, model-only)**, compared with about 16–21 ms in PyTorch. That is a **3–4.5x speed-up from TensorRT**, and it leaves a lot of headroom over the 30 FPS a rehab app needs. A Raspberry Pi 5 on its CPU alone manages only about **15 FPS at best (NCNN)**. A Pi 5 with Hailo-8L runs yolov8s_pose at about **46–65 FPS at batch 1**. Coral cannot run YOLO-pose well and is limited to MoveNet or PoseNet.

### Cited Findings
**Jetson Orin Nano Super (8GB, JetPack 6.2.x)**
- YOLOv8n, 640x640, batch 1, MAXN_SUPER, JetPack 6.2.2 / L4T 36.5.0: PyTorch FP32 20.88 ms (47.9 FPS); TensorRT FP32 8.36 ms (119.5 FPS); TensorRT FP16 4.43 ms (225.3 FPS); TensorRT INT8 3.49 ms (286.2 FPS). Latency is CUDA-event or `trtexec` GPU compute time and excludes host-device transfers and pre/post-processing. — [hokwangchoi/jetson-orin-nano-benchmarks (GitHub)](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
- YOLO26n, 640, coco128, Ultralytics 8.4.33, on the Orin Nano Super Developer Kit: PyTorch 15.60 ms; TorchScript 12.60; ONNX 15.76; TensorRT FP32 7.53; **TensorRT FP16 4.57**; TensorRT INT8 3.80 ms (mAP 0.449 vs 0.480 for FP16). NCNN 47.5 ms and TFLite 286 ms, so the CPU-oriented formats run far slower. Inference only. The device notes say Orin Nano was tested on JP6.1. — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Qengineering's table lists YOLOv8n (detection) at about 100 FPS on Orin Nano and 19 FPS on the original Jetson Nano. — [Qengineering YoloV8-TensorRT-Jetson_Nano (GitHub)](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano) (taken from a search snippet; the conditions behind the number were not verified)
- Real-world caveat: a forum user on Orin Nano 8GB reported only about 10 FPS with YOLOv8 even after converting to TensorRT. This usually means the pipeline bottleneck is Python pre/post-processing, camera I/O or the power mode. — [NVIDIA forum: Slow FPS on Orin Nano 8GB](https://forums.developer.nvidia.com/t/slow-fps-on-orin-nano-8-gb-yolov8/280071). A separate thread reports low YOLO11n TRT FP16 performance on the Super. — [NVIDIA forum: Low Performance – Orin Nano Super](https://forums.developer.nvidia.com/t/low-performance-jetson-orin-nano-super/318862)
- A Hackster project on a 4GB Orin Nano reports that a C++ TensorRT pipeline roughly doubled speed over Python. — [Hackster: Pushing limits YOLOv8 vs v26 on Jetson Orin Nano](https://www.hackster.io/qwe018931/pushing-limits-yolov8-vs-v26-on-jetson-orin-nano-b89267)
- Pose on a larger Jetson, for scale: yolov8s-pose TRT engine, 640, batch 1, at 258.8 FPS on AGX Orin and 40.6 FPS on AGX Xavier. — [the0807/YOLOv8-ONNX-TensorRT (GitHub)](https://github.com/the0807/YOLOv8-ONNX-TensorRT) (from a search snippet)

**Jetson Orin NX 16GB**
- YOLO26n 640: PyTorch 13.90 ms; ONNX 14.18; TRT FP32 7.01; TRT FP16 4.13; TRT INT8 3.49 ms (reComputer J4012, JP6.0/5.1.3, Ultralytics 8.4.33). — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- YOLO11, 640, TensorRT FP16: 5.27 ms on Orin NX 16GB. — [Ultralytics DeepStream on Jetson guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)
- In model-only latency, Orin NX 16GB is only about 10% faster than Orin Nano Super for a nano model (4.13 vs 4.57 ms FP16), based on the two Ultralytics tables above.

**Raspberry Pi 5 (CPU only)**
- YOLO26n, 640, FP32, Ultralytics 8.4.x: PyTorch 299.09 ms; TorchScript 353.2; ONNX 125.99; OpenVINO 104.55; LiteRT 123.3; MNN 91.87; **NCNN 67.03 ms**. YOLO26s with NCNN takes 172.88 ms. Ultralytics recommends NCNN as the fastest format on the Pi. — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- ONNX at 640: YOLO26n 128.42 ms vs YOLO11n 147.20 ms. — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- Arm publishes a YOLO11n-pose INT8 ExecuTorch/XNNPACK build targeted at the Pi 5, but the snippet gave no FPS. — [Arm Hugging Face: yolo11n-pose-int8 executorch raspberrypi5](https://developer.arm.com/ai/models/hugging-face/Arm/yolo11n-pose-int8-xnnpack-executorch-raspberrypi5/yolo11n-pose-int8-pte)

**Raspberry Pi 5 + Hailo-8L (AI Kit / AI HAT+ 13 TOPS)**
- `hailortcli run --batch-size 8`, PCIe Gen 3, 640x640: **yolov8s_pose 123.43 FPS**; yolov8s 127.85; yolov6n 354.07; yolov5n_seg 103.57. Posted June 11, 2024. These are raw runtime figures, not application FPS. — [Hailo Community: Raspberry Pi 5 with Hailo-8L Benchmark](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-benchmark/746)
- At batch 1, Seeed reportedly measured YOLOv8s at 80.7 FPS and pose at 65 FPS. A user measured 54 FPS (YOLOv8s) and **46 FPS (pose)**, and their batch-8 runs matched the published figures. — [Hailo Community thread (via search)](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-expected-performance/746); Seeed's own page shows the results only as chart images: [Seeed wiki: Benchmark on RPi5 and CM4 with AI Kit](https://wiki.seeedstudio.com/benchmark_on_rpi5_and_cm4_running_yolov8s_with_rpi_ai_kit/) (int8, 640x640, batch 1 and 8)
- On a CM4-based reComputer R1000 with Hailo-8L, YOLOv8 pose went from 0.5 FPS on the CPU to 27 FPS with Hailo. — [Seeed wiki: YOLOv8 Pose on reComputer R1000 with Hailo-8L](https://wiki.seeedstudio.com/yolov8_pose_estimation_on_recomputer_r1000_with_hailo_8l/)
- With PCIe Gen 2, yolov8s on CM5 + Hailo-8L fell to 28.59 FPS, against 55.73 FPS on Gen 3. — [Hailo Community: YOLOv8s performance results](https://community.hailo.ai/t/yolov8s-performance-results/18146) (via search snippet)
- Several Hailo forum threads report real-world FPS well below the Model Zoo figures, for example "YOLOv8m 640 only 16 FPS vs 130+ expected" and Hailo-8 on Pi 5 underperforming. — [Hailo Community: Low FPS with YOLOv8m](https://community.hailo.ai/t/low-fps-with-yolov8m-640x640-on-raspberry-pi-5-only-16-fps-vs-130-expected/17313/2); [Hailo Community: Hailo-8 on Pi 5 not as good as official](https://community.hailo.ai/t/the-performance-on-the-raspberry-pi-5-with-the-hailo-8-chip-seems-not-good-as-he-official-results/17473)
- Hailo-8L-compiled HEFs run on Hailo-8 but only at Hailo-8L speed. You need to recompile with `--hw-arch` to use the full chip. — [Hailo Community benchmark thread](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-benchmark/746)

**Google Coral Edge TPU**
- Coral's published pose models, single inference on a USB Accelerator with a desktop host: **MoveNet.SinglePose.Lightning 192x192: 7.1 ms; Thunder 256x256: 13.8 ms; PoseNet MobileNetV1 353x481: 5.8 ms, 481x641: 10.3 ms**. Coral describes these as "not production-quality" demo models. — [Coral pose-estimation models](https://gweb-coral-full.uc.r.appspot.com/models/pose-estimation/)
- MobileNet v1 SSD at 224: 109 ms on a desktop Xeon CPU, 6.5 ms with the USB Accelerator (USB 3), 11 ms on the Dev Board, and 353 ms on a quad A53 CPU. C++ benchmarks, page dated 2020. — [Coral Edge TPU benchmarks](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/benchmarks/)

**Laptop CPU / cloud GPU (reference)**
- Ultralytics pose task table at 640: **YOLO26n-pose CPU ONNX 40.3 ms; T4 TensorRT10 1.8 ms**; mAP-pose 57.2; 2.9M params; 7.6 GFLOPs. The CPU model is not stated. — [Ultralytics Pose task docs](https://docs.ultralytics.com/tasks/pose/)
- Jetson AGX Thor, for scale: YOLO26n TRT FP16 1.39 ms. — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences
- Model-only latency of about 4.5 ms (FP16) on Orin Nano Super means the **end-to-end pipeline will be limited by camera capture, pre/post-processing, keypoint/angle logic and UI**, not by the GPU. A realistic Python pipeline should hold 30 FPS or more at 640x480. The 10 FPS forum report shows this only happens if the pipeline is optimised (TRT engine, MAXN SUPER, no needless CPU copies).
- YOLOv8n-pose costs somewhat more than YOLOv8n detection because of the keypoint head. Its latency on Orin Nano Super with TRT FP16 is probably in the 5–8 ms range, but this is an extrapolation with no direct measurement found.
- A 640x480 input will be letterboxed to 640x640 in Ultralytics, or set imgsz=(480,640), so the 640 benchmarks are a fair upper bound.
- A Pi 5 on its CPU (NCNN, about 15 FPS model-only for YOLO26n) is borderline for 30 FPS rehab feedback. A Pi 5 with Hailo-8L can reach real time with yolov8s_pose, but only after HEF compilation and PCIe Gen 3 setup.
- On Coral, MoveNet Lightning at 7 ms is fast, but it is single-person, 17 keypoints, fixed 192x192 input, and needs a quantized TFLite model. YOLOv8-pose is not a natural fit for the Edge TPU.

### Gaps
- No primary source found with **YOLOv8n-pose** latency on Orin Nano (original or Super), Orin Nano 4GB, or old Jetson Nano. Ultralytics has replaced its YOLOv8/YOLO11 Jetson tables with YOLO26 tables. Earlier versions, such as the Dec 2024 Ultralytics blog "YOLO11 on Jetson Orin Nano Super", show their numbers only as figures that could not be extracted.
- No direct measurement found for Orin Nano 8GB original vs Super with the same YOLO model and conditions. NVIDIA's blog gives 1.4–1.7x Super-mode speed-ups on ViT-family models only (see Q3).
- No laptop GPU (RTX 30/40 mobile) or A10G YOLO-pose figure was found.
- From memory and not re-verified this session: older Ultralytics tables listed YOLOv8n-pose at about 131.8 ms CPU ONNX and 1.18 ms A100 TRT, and YOLO11n-pose at about 52 ms CPU and 1.7 ms T4 TRT. Verify these against the Ultralytics GitHub history before citing them.

---

## Q2. What FPS does MediaPipe Pose / MoveNet achieve on Pi 5, Coral and Jetson?

### Takeaway
I found little reliable data for MediaPipe Pose. MoveNet on Coral is well documented (7–14 ms). On the Pi 5 CPU, MoveNet Lightning reportedly takes about 12 ms. **No primary benchmark of MediaPipe Pose (BlazePose) on the Pi 5 or Jetson was found.** MediaPipe's Python package runs on the CPU on Linux ARM, so on Jetson it does not use the GPU unless you build a custom GPU build. That is a major argument for YOLOv8n-pose with TensorRT on Jetson.

### Cited Findings
- MoveNet Lightning on Coral USB: 7.1 ms (192x192); Thunder: 13.8 ms (256x256). — [Coral pose-estimation models](https://gweb-coral-full.uc.r.appspot.com/models/pose-estimation/)
- On the Pi 5 CPU, one secondary reference gives about 12 ms per inference for MoveNet Lightning and about 30 ms for Thunder. — [applied-ee: Pose Estimation](https://applied-ee.github.io/embedded/docs/edge-ai/computer-vision/pose-estimation/) (from a search snippet; secondary aggregator, not verified)
- A Hackster/element14 project by AlbertaBeef profiled MediaPipe pose, hand and face models accelerated on the Pi 5 AI Kit (Hailo-8L). The pose FPS was not extracted. — [Hackster: Accelerating the MediaPipe models on RPi5 AI Kit](https://hackster.io/AlbertaBeef/accelerating-the-mediapipe-models-on-raspberry-pi-5-ai-kit-1698fe); [Hailo Community thread](https://community.hailo.ai/t/accelerating-the-mediapipe-models-with-hailo/3012)
- On a Pi 4 CPU, popular object detection models "top out at 2-5 fps". Geerling's Pi 5 post (May 30, 2024) does not report Pi 5, Hailo or Coral FPS. It notes that Coral adds about $60 to a build. — [Jeff Geerling: Testing object detection on Pi 5](https://www.jeffgeerling.com/blog/2024/testing-object-detection-yolo-mobilenet-etc-picamera2-on-pi-5/)
- One report of MediaPipe object detection on Pi hardware gives about 2 FPS. — [CNX Software: MediaPipe for Raspberry Pi](https://cnx-software.com/?p=117921) (from a search snippet; board not named)

### Inferences
- MediaPipe Pose Lite/Full runs at 256x256 internally after the person detector. It is therefore much cheaper than a 640-input YOLO, and reaching 15–30 FPS on a Pi 5 CPU is plausible. This is unverified.
- For a rehab app on Jetson, MediaPipe gains little from the 67-TOPS GPU. YOLOv8n-pose with TensorRT is the path that makes use of the hardware being justified.

### Gaps
- No primary-source FPS for MediaPipe Pose on the Pi 5, Jetson Orin Nano or a laptop CPU.
- No MoveNet TensorRT FPS on Jetson.

---

## Q3. Power consumption under load and perf-per-watt

### Takeaway
Orin Nano 8GB runs in configurable **7/15 W (original) or 15/25 W/MAXN SUPER (JetPack 6.2)** modes. The Coral Edge TPU is about 2 W (2 TOPS/W). No measured wall-power figures for YOLO pose workloads were found for any platform, so perf-per-watt can only be estimated from rated envelopes.

### Cited Findings
- Orin Nano 8GB power modes: original 7 W and 15 W; Super adds 15 W, 25 W and MAXN SUPER. Orin Nano 4GB: original 7 W and 10 W; Super 10 W, 25 W and MAXN SUPER (NVIDIA's own tables disagree on the 4GB list). Orin NX 16GB: up to 40 W and MAXN SUPER. — [NVIDIA blog: JetPack 6.2 Super Mode](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules)
- Super Mode speed-ups at FP16 TensorRT for the Orin Nano 8GB: clip-vit-base 1.60–1.69x; DINOv2 1.68x; SAM2 1.43x; Grounding DINO 1.52x. NVIDIA gives no YOLO or PeopleNet numbers. — [NVIDIA blog: JetPack 6.2 Super Mode](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules)
- The Edge TPU delivers 4 TOPS at 0.5 W/TOPS, which is 2 TOPS/W and about 2 W. — [Coral Edge TPU benchmarks](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/benchmarks/)
- The 67 TOPS figure is a sparse rating. Dense INT8 is 33 TOPS. — [NVIDIA blog: JetPack 6.2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-launches-new-usd249-ai-development-board-that-does-67-tops)

### Inferences
- Rough perf-per-watt for YOLO-nano at 640 on Orin Nano Super is about 225 FPS (FP16, model-only) / 25 W ≈ **9 FPS/W at the module level**. That is optimistic, because 25 W is a mode cap and not a measured draw. On the Pi 5 CPU, about 15 FPS (NCNN) at a Pi 5 load of roughly 8–10 W (unverified) gives under 2 FPS/W. The Hailo-8L at about 46–65 FPS batch-1 pose likely sits at a few watts on top of the Pi, which makes it competitive per watt.
- For a mains-powered clinic or home rehab kiosk, power is not a binding constraint. Thermal behaviour and a sustained 25 W mode matter more.

### Gaps
- No measured board power (`tegrastats` / INA readings) for YOLO pose on any Jetson. The hokwangchoi repo does not report power.
- Raspberry Pi 5 load power, Hailo-8 / 8L module power, laptop and cloud GPU TDPs were not found in this session's sources. Commonly quoted values, all unverified here: Pi 5 about 5–12 W, Hailo-8 about 2.5 W typical, T4 70 W, A10G 150 W.

---

## Q4. Software ecosystem maturity (CUDA/TensorRT vs Hailo SDK vs Edge TPU)

### Takeaway
Jetson has the most mature path: Ultralytics supports it natively, with one-line `export(format="engine")` to TensorRT FP16/INT8, plus DeepStream, Docker images and Python 3.10 on JetPack 6.x. Hailo needs an offline **HEF compile with the Dataflow Compiler on an x86 host**, its batch-1 performance is sensitive to the pipeline, and forum reports of under-performance are common. Coral needs **fully int8-quantized TFLite** compiled with the Edge TPU compiler, which effectively limits it to MoveNet and PoseNet.

### Cited Findings
- The Ultralytics Jetson guide covers JetPack 4, 5, 6 and 7. Python is 3.8 on JP5, 3.10 on JP6.1 and 3.12 on JP7.2. TensorRT FP32/FP16/INT8 export and DLA are supported, though DLA needs TensorRT ≤ 10.7. — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- PyTorch/TorchVision compatibility problems on JetPack 6.x / CUDA 12.6 are a recurring forum topic. — [NVIDIA forum: Orin Nano issues running YOLO due to PyTorch/TorchVision](https://forums.developer.nvidia.com/t/jetson-orin-nano-issues-running-yolo-ultralytics-due-to-pytorch-torchvision-compatibility-jetpack-6-x-cuda-12-6/365683)
- Super Mode reaches existing Orin Nano kits through a JetPack 6.2 software update. — [NVIDIA blog: JetPack 6.2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules)
- Hailo models are compiled to HEF per architecture (`hailomz compile ... --hw-arch hailo8l`). The Pi 5 link is single-lane PCIe, and some dual-HAT boards drop to Gen 2. — [Hailo Community benchmark thread](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-benchmark/746)
- The AI HAT+ uses the Pi 5's PCIe Gen 3 interface and is backward compatible with the AI Kit. Hailo-8L models run on Hailo-8. — [Raspberry Pi news: AI HAT+](https://www.raspberrypi.com/news/raspberry-pi-ai-hat/); [Tom's Hardware: AI HAT+ 13/26 TOPS](https://tomshardware.com/raspberry-pi/raspberry-pi-release-higher-performance-ai-hat-13-and-26-tops-variants)
- Ultralytics LiteRT (TFLite) export is not supported on the Pi itself, so you export on x86 or macOS. — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- Coral pose models ship as quantized (`quant` / `ptq`) TFLite models "compiled for the Edge TPU". — [Coral pose models](https://gweb-coral-full.uc.r.appspot.com/models/pose-estimation/); the Coral benchmark page is dated ©2020. — [Coral benchmarks](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/benchmarks/)

### Inferences
- Coral's documentation and model zoo appear frozen around 2020–2021, with no new models. It is a weak choice for a 2026 project.
- For a student or clinical prototype, Jetson's single toolchain (PyTorch to ONNX to TensorRT, all on-device) is the lowest-risk option. Hailo is viable but adds an x86 compile step and pipeline tuning.

### Gaps
- No source collected that formally documents Edge TPU compiler operator limits for YOLOv8-pose (full-int8 requirement, unsupported ops falling back to the CPU). This is widely known but was not cited this session.
- Hailo Dataflow Compiler host requirements (x86 Ubuntu, GPU recommended) were not verified.

---

## Q5. Is the old Jetson Nano (JetPack 4.6, Python 3.6, Ubuntu 18.04) still supported?

### Takeaway
**No.** JetPack 4 is end-of-life. Its final release is **4.6.6**, and NVIDIA no longer provides security updates or bug fixes. It is based on Ubuntu 18.04 with Python 3.6, which current Ultralytics cannot run without containers or source builds. Canonical offers paid ESM security updates for the userspace until 2028.

### Cited Findings
- NVIDIA announced End of Life for JetPack 4 with the release of JetPack 4.6.6. — [NVIDIA forum: EOL for JetPack 4 with 4.6.6](https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314301); [JetPack 4.6.6 page](https://developer.nvidia.com/jetpack-sdk-466)
- JetPack 4 is built on Ubuntu 18.04 LTS and is no longer maintained by NVIDIA, so it gets no security updates or bug fixes. Ubuntu Pro ESM provides updates until 2028. — [Canonical/Ubuntu blog: JetPack 4 EOL](https://ubuntu.com/blog/jetpack-4-eol)
- Ultralytics requires Python ≥ 3.7, and pip install fails on JetPack 4's Python 3.6.9. NVIDIA provides PyTorch wheels only for Python 3.6 on JP4. — [NVIDIA forum: Unable to install ultralytics in Python 3.6](https://forums.developer.nvidia.com/t/unable-to-install-ultralytics-in-python3-6-in-jetpack4-6/275989); [NVIDIA forum: Ultralytics on Nano with Python 3.6](https://forums.developer.nvidia.com/t/is-ultralytics-library-compatible-with-jetson-nano-dev-kit-with-default-python-3-6/345326)
- Ultralytics still lists JetPack 4 as supported for the Nano, TX2 and Xavier through a **JetPack 4 Docker image** (tested on JP4.6.1). — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- YOLOv8n reaches about 19 FPS on the original Nano vs about 100 FPS on Orin Nano with TensorRT C++. — [Qengineering YoloV8-TensorRT-Jetson_Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano)

### Inferences
- Pairing an EOL OS (no security patches) with a medical or rehab device that handles patient video is a compliance and security liability. Together with roughly 5x lower YOLO throughput, this rules out the old Nano for new deployments.

### Gaps
- Exact old Jetson Nano specs (472 GFLOPS FP16, 128-core Maxwell, 5/10 W) and current pricing were not re-verified this session.

---

## Pricing notes (for the table)

### Cited Findings
- Orin Nano Super Developer Kit: **$249** (Dec 2024), down from $499 for the original 40-TOPS kit. — [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-launches-new-usd249-ai-development-board-that-does-67-tops); [CNX Software](https://www.cnx-software.com/2024/12/18/249-nvidia-jetson-orin-nano-super-developer-kit-targets-generative-ai-applications-at-the-edge/?amp=1&noamp=mobile). SparkFun warns that tariffs and component costs may change the price. — [SparkFun](https://www.sparkfun.com/nvidia-jetson-orin-nano-developer-kit.html)
- AI HAT+: **$70 (13 TOPS Hailo-8L) and $110 (26 TOPS Hailo-8)** at the October 2024 launch. SparkFun currently lists the 26 TOPS board at $132. — [Raspberry Pi news](https://www.raspberrypi.com/news/raspberry-pi-ai-hat/); [SparkFun 26 TOPS](https://www.sparkfun.com/raspberry-pi-ai-hat-26-tops.html); [SparkFun 13 TOPS](https://www.sparkfun.com/raspberry-pi-ai-hat-13-tops.html)
- Coral USB Accelerator adds about $60. — [Jeff Geerling](https://www.jeffgeerling.com/blog/2024/testing-object-detection-yolo-mobilenet-etc-picamera2-on-pi-5/)

### Inferences
- Approximate INR at ₹85/USD, before import duty and GST: Orin Nano Super ≈ ₹21,200; original Orin Nano kit ≈ ₹42,400; AI HAT+ 13T ≈ ₹6,000; AI HAT+ 26T ≈ ₹9,400; Coral USB ≈ ₹5,100. Indian retail (for example from authorised NVIDIA distributors) is typically noticeably higher. Check local listings before quoting.
- **Pi 5 8GB + AI HAT+ 26T + PSU and cooler** probably totals roughly $200. That is close to the Orin Nano Super's $249, which has 8 GB of unified memory, a CUDA GPU, and native TensorRT support for YOLOv8n-pose. This is the main cost-justification argument.

### Gaps
- No cited price was collected for the Raspberry Pi 5 (4/8 GB), Coral Dev Board, Orin Nano 4GB, Orin NX modules or kits, or the old Jetson Nano. No cloud T4/A10G hourly rates or actual Indian retail prices were collected either.
