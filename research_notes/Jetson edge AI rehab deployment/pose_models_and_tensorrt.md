# Pose-Estimation Model Comparison and TensorRT Precision Trade-offs for a Rehab Joint-Angle App (Jetson Orin Nano)

Scope note: the app uses 12 body keypoints (shoulders, elbows, wrists, hips, knees, ankles) for elbow, knee and hip angles. Both MediaPipe (33 landmarks) and COCO-17 models (YOLO, MoveNet, RTMPose, ViTPose) cover all 12. Research was capped at about 15 tool calls, so several numbers come from search-result summaries rather than full-text reads. These are marked "(search summary)" and should be checked before they are quoted in formal coursework.

## Q1. Model comparison: keypoints, 2D/3D, single vs multi-person, accuracy, size, speed

### Takeaway
For a single-patient rehab app on a Jetson Orin Nano, the choice comes down to two models. MediaPipe BlazePose (Full) gives 33 landmarks and metric 3D "world" coordinates, which reduces perspective error in angles. YOLO-pose (v8n/11n/26n) is a multi-person, 2D-only COCO-17 model that becomes very fast after a TensorRT FP16 export. Each source quotes accuracy on a different benchmark (COCO mAP, PCK@0.2 on Google's internal sets, single-person COCO subsets), so the accuracy figures below cannot be compared directly across model families.

### Cited Findings

**Ultralytics YOLO-pose (COCO-17, 2D, multi-person, one-stage). All at 640 px, COCO val2017**
- YOLOv8-pose: n 50.4 mAPpose50-95 / 80.1 mAP50, 3.3M params, 9.2 GFLOPs, CPU ONNX 131.8 ms, A100 TensorRT 1.18 ms. s 60.0 / 86.2, 11.6M. m 65.0 / 88.8, 26.4M. l 67.6. x 69.2. x-p6 (1280 px) 71.6. The CPU model is not specified. — [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/)
- YOLO11-pose: n 50.0 / 81.0, 2.9M params, 7.5 GFLOPs, CPU ONNX 52.4 ms, T4 TensorRT10 1.7 ms. s 58.9 / 86.3, 9.9M, T4 2.6 ms. m 64.9 / 89.4, 20.9M, T4 4.9 ms. l 66.1. x 69.5. — [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- YOLO26-pose (newest Ultralytics generation, NMS-free end-to-end): n 57.2 / 83.3, 2.9M params, 7.6 GFLOPs, CPU ONNX 40.3 ms, T4 TensorRT10 1.8 ms. s 63.0 / 86.6. m 68.8 / 89.6. l 70.4. x 71.6. Measured on COCO Keypoints val2017, batch=1, nms=False. — [Ultralytics Pose task docs](https://docs.ultralytics.com/tasks/pose/)
- YOLO26n-pose has about the same parameter count as YOLOv8n-pose, with +6.8 mAPpose50-95 (57.2 vs 50.4). — computed from the two Ultralytics tables above.

**MediaPipe BlazePose / Pose Landmarker (33 landmarks, 2D image coords plus 3D world coords; single-person top-down pipeline of detector then landmark tracker)**
- Variants Lite / Full / Heavy. PCK@0.2 on Google's Yoga/Dance/HIIT evaluation sets: Heavy about 96.4–97.5, Full about 95.5–96.3, Lite about 90.2–93.5. — [MediaPipe pose docs (GitHub)](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) (search summary)
- Latency on Pixel 3 TFLite GPU: Heavy 53 ms, Full 25 ms, Lite 20 ms. On a 2017 MacBook Pro: 38 / 27 / 25 ms. — [MediaPipe pose docs (GitHub)](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) (search summary)
- Model card: Lite about 44 FPS on Pixel 3 CPU and about 49 FPS on its GPU. Full about 18 FPS on CPU and about 40 FPS on GPU. File sizes about 3 MB (Lite), 6 MB (Full), 26 MB (Heavy). — [BlazePose GHUM 3D model card](https://dcmpx.remotevs.com/com/googleapis/storage/SL/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) (search summary, mirror of Google's PDF)
- The original BlazePose paper reports a human-annotator baseline of 97.2 PCK@0.2. — [BlazePose paper, arXiv 2006.10204](https://arxiv.org/pdf/2006.10204) (search summary)
- PCK@0.2 on Google's own sets is not COCO mAP. Google also reports mAP in the same table (Yoga, Lite to Heavy: 45.0 to 68.1). — [MediaPipe pose docs](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) (search summary)

**MoveNet (Google, 17 COCO keypoints, 2D, SinglePose variants)**
- Lightning targets latency and Thunder targets accuracy. mAP on a single-person-cropped COCO subset: Thunder FP16 72.0, Thunder INT8 68.9, Lightning FP16 63.0, Lightning INT8 57.4. — [TensorFlow Lite pose estimation overview](https://www.tensorflow.org/lite/examples/pose_estimation/overview) (search summary)
- Latency (end-to-end, TF 2.5 sample apps): Thunder FP16 155 ms Pixel 5 CPU (4 threads), 45 ms Pixel 5 GPU, 594 ms Raspberry Pi 4. Lightning FP16 60 / 25 / 186 ms. Thunder INT8 100 / 52 / 251 ms. — [TensorFlow Lite pose estimation overview](https://www.tensorflow.org/lite/examples/pose_estimation/overview); [TF blog, Aug 2021](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) (search summary)
- MoveNet mAP is measured on single-person crops, so it is not comparable with YOLO's full-image multi-person COCO mAP. — same source.

**RTMPose (MMPose, top-down, needs a person detector)**
- RTMPose-m: 75.8% COCO AP, 90+ FPS on an Intel i7-11700 CPU, 430+ FPS on a GTX 1660 Ti GPU. RTMPose-s: 72.2% AP at 70+ FPS on a Snapdragon 865 phone. RTMPose-l: 67.0% AP on COCO-WholeBody (130+ FPS). — [RTMPose, arXiv 2303.07399](https://arxiv.org/abs/2303.07399)
- Top-down FPS figures usually exclude the person-detector stage. The abstract does not state this explicitly (see Gaps).

**ViTPose (top-down, Vision Transformer backbone)**
- ViTPose-B 75.8 AP and ViTPose-L 78.3 AP on COCO val. The best single model (ViTAE-G backbone) reaches 80.9 AP on COCO test-dev. Different splits, so do not compare directly. — [ViTPose, arXiv 2204.12484](https://arxiv.org/pdf/2204.12484) (search summary)

**OpenPose (CMU, bottom-up, Part Affinity Fields)**
- Bottom-up multi-person method using PAFs. "Realtime performance regardless of the number of people." First combined body+foot detector. The release covers body, foot, hand and face keypoints. — [Cao et al., OpenPose, arXiv 1812.08008](https://arxiv.org/abs/1812.08008)
- The abstract page gives no COCO AP or timing numbers. A commonly quoted "61.8 AP" figure could not be verified (see Gaps).

### Inferences
- **For the 12-keypoint rehab use case, every model listed has the needed joints.** The deciding factors are 3D output, single vs multi-person, and Jetson throughput.
- **MediaPipe Pose Landmarker (Full) is the better default for clinical-style angle measurement of one patient.** Its world landmarks allow 3D angle computation, which is less sensitive to camera perspective (see Q2). Its tracking pipeline is built for one person in frame. Heavy improves PCK by about 1 point over Full at roughly 2x latency on Pixel 3 GPU. Lite loses about 3–5 PCK points.
- **YOLO-pose (n) is the better choice when** several people may be in frame (therapist plus patient), when GPU throughput via TensorRT is the priority, or when students want to fine-tune on custom data. It gives only 2D image-plane angles. For a new build, YOLO11n-pose or YOLO26n-pose is a drop-in upgrade over YOLOv8n-pose: same API and size class, with higher mAP (YOLO26n: +6.8).
- **RTMPose and ViTPose have higher COCO AP but need a separate detector stage and the MMPose toolchain,** which makes them heavier to deploy on Jetson. They are good "accuracy ceiling" references for student comparisons. ViTPose-B or larger is unlikely to be real-time on an Orin Nano (inference only: no Jetson numbers were found).
- **OpenPose is largely of historical and teaching interest now.** It is a heavy bottom-up network, and the newer models above outperform it.
- **MoveNet Lightning is a good CPU or Raspberry Pi-class baseline.** On a Jetson GPU it offers little over YOLO-n with TensorRT.

### Gaps
- No verified OpenPose COCO AP or Jetson FPS figures. The full TPAMI paper was not retrieved.
- No first-party Jetson Orin Nano FPS numbers for MediaPipe Pose Landmarker. MediaPipe's Python package runs mostly on CPU on Jetson/Linux-ARM, with GPU delegate support limited, so measured FPS should come from students' own benchmarks. This is unverified and not sourced here.
- RTMPose end-to-end FPS including the detector, and RTMPose/ViTPose numbers on Jetson, were not found.
- The MoveNet Lightning INT8 latency row was truncated in the search results.

## Q2. Accuracy of camera-based joint angles vs gold standard (goniometer / marker-based mocap)

### Takeaway
Single-camera pose estimation typically agrees with marker-based motion capture to within about 4–8° for sagittal-plane hip and knee angles in walking, and to roughly 3–9° in cycling. Errors grow sharply with off-axis camera views, frontal-plane angles, occlusion (especially the elbow), and unstable tasks. Static goniometer comparisons with MediaPipe have shown ICC of about 0.87 with an RMSE of about 15° in one recent preprint. Camera-based angles should therefore be presented as "screening / trend" measures, not goniometer replacements, unless they are validated in the app's own setup.

### Cited Findings
- **OpenPose 2D gait vs 3D mocap (Stenum, Rossi & Roemmich, PLOS Comput Biol 2021):** healthy adults walking with a sagittal camera. MAE was hip 4.0°, knee 5.6°, ankle 7.4°. Temporal gait parameters were within 0.01 s and step length within 0.018 m. — [Stenum et al. 2021 (PMC8099131)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8099131) (search summary)
- **Smartphone pose estimation vs 3D motion analysis:** MAE was ankle 7.13°, knee 5.82°, hip 7.73°. Other OpenPose studies found good knee/ankle peak-angle agreement but poor or no correlation for the hip. — [Sensors 2023, 23(24):9799](https://www.mdpi.com/1424-8220/23/24/9799); [PMC8512754](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754) (search summary)
- **MediaPipe/BlazePose knee vs goniometer and Kinovea (JMIR preprint, 2026, not peer-reviewed):** At static goniometer-set positions (30/60/90/120°), ICC was 0.874 and RMSE about 15.5°, with error depending on the angle. In dynamic tasks against Kinovea 2D annotation, overall ICC was 0.982, but MediaPipe systematically underestimated knee angle by about 5°. Single-leg stance ICC was 0.136. With all outputs included (no detection-quality filtering), squat ICC fell to 0.428 and RMSE rose to about 25°. — [JMIR preprint 102399](https://preprints.jmir.org/preprint/102399) (search summary)
- **Cycling (Sport Sciences for Health, 2024):** MediaPipe typical errors were about 3–9° against marker-based digitisation. Correlations were weaker than for TransPose. — [Springer s11332-023-01075-7](https://link.springer.com/content/pdf/10.1007/s11332-023-01075-7.pdf) (search summary)
- **Knee valgus, frontal plane (Heliyon 2024):** MediaPipe absolute angles did not show significant concurrent validity against 3D Vicon. Change-from-initial-contact values did better. — [Heliyon 2024 (DOAJ)](https://doaj.org/article/2d283f6411874c82988f54dd2e60829f) (search summary)
- **RGB/RGB-D body tracking vs goniometer (Sensors, Dec 2022):** In one comparison, MediaPipe's error was within the error range of short- and long-arm goniometers. — [Sensors 2023, 23(1):3 / PMC9824231](https://www.mdpi.com/1424-8220/23/1/3) (search summary; full text blocked)
- **Camera viewing angle, OpenPose (Sensors 2025, 25(3):799):** Joint angles from four camera viewing angles were compared with marker-based capture, including the elbow. There were significant biases between viewing angles, and back-facing cameras gave the lowest percentage deviations. — [MDPI Sensors 25(3):799](https://mdpi.com/1424-8220/25/3/799); [PMC11819822](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11819822/) (search summary; full text returned 403)
- **Elbow flexion, BlazePose vs IMUs vs optical (ETH Zurich, 2025):** Accuracy depended significantly on participant orientation to the camera. Bland-Altman analysis showed non-constant (proportional) error for pose estimation, which underestimated large elbow flexion angles. IMUs showed mostly constant offsets. — [ETH Research Collection](https://www.research-collection.ethz.ch/items/62b944dd-2f46-401c-acdc-b9d98ee4d441) (search summary)
- **Monocular 3D markerless (IEEE Access 2025):** Elbow was most accurate with the camera placed at 90° and 135° around the subject. The authors propose a distance-dependent tilt-correction ratio for MediaPipe. — [IEEE Access 2025 (DOAJ)](https://doaj.org/article/bbd796d1b42c43c994aa24a51ec1ec64) (search summary)
- **OpenPose in sports validation (Sports Engineering 2024):** Elbow angles had the largest errors, attributed to occlusion and systematic landmark differences between pose estimation and marker-based capture. — [Springer s12283-024-00460-w](https://link.springer.com/article/10.1007/s12283-024-00460-w) (search summary)

### Inferences
- **Practical accuracy budget for the app:** in a well-controlled sagittal setup, expect roughly 4–8° error at the knee and hip. Elbow error is likely larger and depends on view. Off-axis or frontal-plane measurements can be much worse (ICC can collapse).
- **2D vs 3D:** a 2D image-plane angle equals the true joint angle only when the limb segments lie in a plane parallel to the image sensor. Rotating the limb out of that plane foreshortens segments and distorts the angle. This is consistent with the strong viewpoint dependence reported above. MediaPipe world landmarks reduce this geometrically, but they are a monocular estimate of depth, and the BlazePose elbow study still found orientation-dependent error.
- **Recommended protocol for students:** fix the camera perpendicular to the plane of motion (side-on for knee/hip flexion and elbow flexion) at joint height, about 2–3 m away, with the full body in frame. Use good, even lighting and fitted clothing. Use one camera per plane. Use the landmark visibility/confidence score to reject frames. Validate the setup against a goniometer at a few fixed angles and report bias plus limits of agreement (Bland-Altman) and ICC.
- **Proportional error:** the underestimation at large flexion means a constant offset correction will not fully fix the error. A per-joint linear calibration against a goniometer is a reasonable student project.

### Gaps
- No systematic review with pooled error across studies was retrieved.
- Effects of clothing and lighting were not quantified in the retrieved sources; they are commonly cited, but no numbers were found here.
- No direct head-to-head of MediaPipe vs YOLO-pose joint-angle accuracy against mocap was found.
- Several full texts (MDPI/PMC) were blocked (403 or bot check), so the per-joint numbers for the viewing-angle study are not available.

## Q3. TensorRT precision: FP32 vs FP16 vs INT8 on Jetson Orin; Ultralytics export benchmarks

### Takeaway
On a Jetson Orin Nano Super, TensorRT FP16 roughly halves YOLO-nano latency against TensorRT FP32 (3.4x faster than PyTorch) with no measurable accuracy loss. INT8 adds only a small further gain at the nano size and can severely damage pose accuracy: Ultralytics measured YOLO26n-pose falling from 56.7 to 33.4 pose mAP in INT8. **FP16 is the right setting for a pose app; avoid INT8 for keypoints unless re-validated with a large calibration set or QAT.**

### Cited Findings
- **Ultralytics Jetson Orin Nano Super benchmark (detection, imgsz 640, Ultralytics 8.4.33, inference-only time excluding pre/post-processing; mAP likely on coco128).** YOLO26n results (mAP50-95 / ms):
  - PyTorch 0.479 / 15.60
  - TorchScript 0.477 / 12.60
  - ONNX 0.476 / 15.76
  - TensorRT FP32 0.477 / 7.53
  - TensorRT FP16 0.480 / 4.57
  - TensorRT INT8 0.449 / 3.80

  YOLO26s: FP32 13.60 ms, FP16 7.17, INT8 5.25 (mAP 0.566 / 0.565 / 0.547). YOLO26m: FP32 28.19, FP16 13.59, INT8 9.30 (mAP 0.622 / 0.623 / 0.582). — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- The same guide advises MAX power mode (`sudo nvpmodel -m 0`), `sudo jetson_clocks`, monitoring with jtop, headless operation, and calibrating INT8 on the target device. Its DLA table does not list the Orin Nano (the Orin Nano has no DLA cores). — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- **Ultralytics TensorRT integration (RTX PRO 6000 Blackwell, TensorRT 11.4):** FP16 kept FP32 accuracy within 0.1 points on every model. INT8 lost 1.7–2.9 mAP on YOLO26n–l detection. **YOLO26n-pose: PyTorch 57.0, FP32 56.7, FP16 56.7, INT8 33.4 pose mAP50-95.** Latency for YOLO26n-pose: PyTorch 3.39 ms, FP32 0.76, FP16 0.51, INT8 0.59, so INT8 was slower than FP16 here. INT8 beat FP16 latency only for the larger models (l, x) and for the classifier. INT8 can shift confidence values, so choose thresholds from the INT8 engine's own F1 curve. — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/)
- **INT8 calibration requirements:** `quantize=8` runs post-training quantization using the `data` YAML's val split. NVIDIA recommends at least 500 representative images, and the export warns below 300. Without `data`, the export falls back to coco8 (4 images), which is "only enough to test the export". No calibration cache is reused. TensorRT 7–10 uses an implicit MINMAX calibrator on GPU. TensorRT 11 uses ModelOpt explicit quantization with entropy calibration on about 512 images. QAT checkpoints skip calibration. — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/)
- **Engine portability:** engines are specific to the GPU model and the TensorRT version. Build on the Jetson itself and rebuild after any JetPack or TensorRT upgrade. Export requires a GPU. TensorRT 10.2.0 is unsupported. — [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/)

### Inferences
- **Speedups on Orin Nano Super:** FP16 vs FP32 is about 1.65x for YOLO26n and about 2x for s/m. FP16 vs PyTorch is about 3.4x for n. INT8 vs FP16 is only about 1.2x at nano size.
- **Why INT8 hurts pose so much:** keypoint regression needs sub-pixel coordinate precision, which coarse INT8 activation ranges destroy more than box or class outputs. This is an interpretation consistent with the 56.7→33.4 result, not a sourced explanation.
- **Joint angles are especially sensitive:** a few pixels of keypoint error on a short forearm segment changes the angle by several degrees. That makes FP16 the safe choice for the app.
- **These benchmarks are YOLO26 detection and pose models, not YOLOv8n-pose on Jetson.** The FP32→FP16 latency ratio should transfer roughly to YOLOv8n-pose and YOLO11n-pose (similar architecture and FLOPs), but students should run `yolo benchmark model=yolov8n-pose.pt data=coco8-pose.yaml imgsz=640 half=True device=0` on their own board to get real numbers.

### Gaps
- No Ultralytics Jetson table specifically for pose models (YOLOv8n-pose or YOLO11n-pose) was found. The Jetson tables are for detection.
- The precise dataset behind the Jetson mAP numbers is not explicitly stated (the reproduction command suggests coco128, a very small set). Small mAP differences between formats there are noise.
- No NVIDIA TensorRT primary documentation was fetched within the tool budget. Calibration details come via Ultralytics.

## Q4. Smoothing / filtering of jittery keypoints: EMA vs One-Euro vs Kalman

### Takeaway
A fixed-alpha EMA trades jitter for lag uniformly: strong smoothing makes fast reps lag. The One-Euro filter (Casiez et al., CHI 2012) adapts its cutoff to speed, smoothing heavily when the joint is still and lightly when it moves. That is a better fit for rehab reps, where both steady holds and fast movements matter. Kalman filters add a motion model and are more complex to tune. No retrieved source quantitatively compares them on pose keypoints.

### Cited Findings
- The 1€ filter is "a simple speed-based low-pass filter." A low cutoff reduces jitter at low speeds, and the cutoff increases with speed to reduce lag. In the authors' comparison with other filters it had less lag at a reference level of jitter reduction. — [Casiez, Roussel & Vogel, CHI 2012 (HAL)](https://hal.archives-ouvertes.fr/hal-00670496); [1€ filter project page](https://cristal.univ-lille.fr/~casiez/1euro)
- Tuning procedure from the authors: start with beta = 0 and min_cutoff of about 1 Hz. Lower min_cutoff if there is jitter at slow speed, and raise beta if there is lag at high speed. — [1€ filter project page](https://cristal.univ-lille.fr/~casiez/1euro)
- Kalman filters denoise joint positions by weighting kinematic (motion-model) predictions against observations. The 1€ filter is described as adaptively balancing jitter reduction with low latency. — [arXiv 2602.21702](https://arxiv.org/pdf/2602.21702) (search summary)

### Inferences
- **EMA latency:** EMA y_t = αx_t + (1−α)y_{t−1} has an effective time constant of about (1/α − 1) frames. At 30 FPS, α = 0.3 gives roughly 2.3 frames, or about 78 ms, of lag. This is standard filter math, not from a retrieved source.
- **Where to filter:** smoothing the keypoints (x, y[, z]) with One-Euro before computing the angle generally preserves geometry better than smoothing the angle alone. Smoothing the final angle is simpler and is fine for display. Both are valid student experiments.
- **Kalman:** a constant-velocity Kalman filter can predict through brief dropouts and occlusions, which EMA and One-Euro cannot do. It needs process/measurement noise tuning and adds code complexity.
- **Free smoothing in MediaPipe:** the Pose Landmarker in VIDEO/LIVE_STREAM mode already applies internal temporal smoothing and tracking, so extra smoothing on top may add unnecessary lag. YOLO-pose is per-frame, so it benefits more from external filtering. (Inference about MediaPipe internals; not verified against current docs in this session.)

### Gaps
- No quantitative benchmark comparing EMA, One-Euro and Kalman on 2D/3D pose keypoints (jitter vs latency in ms) was found.
- Current MediaPipe Tasks documentation on its internal smoothing was not fetched.
