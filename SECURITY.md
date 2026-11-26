# Security & Privacy Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
[GitHub private vulnerability reporting](https://github.com/Achyut0617/rehab-motion-evaluator/security/advisories/new).
You can expect an initial response within a few days.

## Secrets

- API keys (e.g. `GEMINI_API_KEY`) belong only in your local `.env`, which is git-ignored.
- `.env.example` must always contain empty placeholder values.
- If a key is ever committed by mistake, **revoke it immediately** at
  <https://aistudio.google.com/apikey>; deleting the commit is not enough.

## Patient data

This project is designed to keep health data on the edge device:

- No raw video is stored or uploaded; only joint angles and session metrics are logged.
- Session data lives in a local SQLite database (`data/rehab_sessions.db`) and CSV exports
  (`data/exports/`), both git-ignored.
- The optional AI coach sends only numeric metrics to the LLM; patient names are never sent.

## Disclaimer

This is a research prototype, **not a medical device**. It must not be used for diagnosis
or treatment decisions without supervision by a qualified clinician.
