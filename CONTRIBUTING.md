# Contributing

Thanks for your interest in improving the Rehab Motion Evaluator!

## Development setup

```bash
git clone https://github.com/Achyut0617/rehab-motion-evaluator.git
cd rehab-motion-evaluator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # lightweight: tests + lint, no ML runtime
cp .env.example .env
```

For the full app (MediaPipe / YOLOv8-pose), install `requirements.txt` instead.
Demo mode (`CAMERA_SOURCE=demo`, `POSE_BACKEND=demo`) runs without a camera or GPU.

## Before opening a pull request

```bash
ruff check .
pytest
```

Both run automatically in CI on every push and pull request.

## Guidelines

- Keep pull requests focused: one feature or fix per PR.
- Add or update tests in `tests/` for any change to angle math, rep counting, evaluation or session logic.
- New exercises go in `config/exercises.json`; thresholds should be reviewed by a physiotherapist.
- Use [Conventional Commits](https://www.conventionalcommits.org/) style messages
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`).
- **Never commit `.env`, API keys, patient data, the SQLite database or session exports.**
- No raw video may ever be stored or transmitted; patient names must never be sent to the LLM.
