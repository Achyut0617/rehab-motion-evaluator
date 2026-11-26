# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-26

### Added
- Real-time pose estimation with MediaPipe (3D world landmarks), YOLOv8-Pose and FP16 TensorRT backends.
- Rep counting and form feedback for knee extension, bicep curl and sit-to-stand, with left/right/both side switching and 3-second calibration.
- Flask dashboard with offline Chart.js, live `tegrastats` hardware telemetry and multilingual (EN/HI/MR) voice cues.
- Patient profiles, per-rep analytics, cross-session progress and physio-approved adaptive targets.
- Local SQLite session storage with per-frame telemetry and CSV export (no raw video stored).
- Optional Gemini AI session reports and physio assistant with offline template fallback.
- Jetson setup script, TensorRT export and benchmarking scripts, deployment research notes.
- Project tooling: `pyproject.toml`, ruff lint config, `requirements-dev.txt`, GitHub Actions CI, Dependabot.
- Community files: MIT license, contributing guide, security & privacy policy, issue and PR templates.

[1.0.0]: https://github.com/Achyut0617/rehab-motion-evaluator/releases/tag/v1.0.0
