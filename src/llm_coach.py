"""AI coaching layer using the free Google Gemini API (with an offline template fallback).

Design rules:
  * The LLM is never in the real-time loop — it runs only after a session, on demand.
  * It only explains numbers computed by src/progress.py; it never diagnoses or
    changes thresholds (target changes require physiotherapist approval).
  * No patient name is sent to the API — only age, condition and movement metrics.
  * If no API key is set or the call fails, a deterministic offline report is returned.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Fast "lite" models first (1-3 s on the free tier); larger models as fallbacks.
FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
)

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi (Devanagari script)", "mr": "Marathi (Devanagari script)"}

SYSTEM_PROMPT = """You are a supportive physiotherapy exercise assistant inside a rehabilitation \
monitoring prototype. You receive movement metrics measured by a pose-estimation system.

Rules:
- Only interpret the metrics provided. Never invent numbers.
- Do NOT diagnose, prescribe medication, or claim clinical validity. Scores are computational indicators.
- If metrics suggest pain risk (very fast/jerky reps, big fatigue drop, large left/right gap), advise \
the patient to consult their physiotherapist rather than pushing harder.
- Threshold/target changes are only suggestions for the physiotherapist to approve.
- Patient-facing text: warm, simple, encouraging, short sentences, suitable for elderly patients.
- Clinician-facing text: concise, factual, uses degrees/seconds, may stay in English terms for metrics."""


class LLMCoach:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 25.0,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("GEMINI_MODEL", "") or FALLBACK_MODELS[0]).strip()
        self.preferred_model = self.model
        self.timeout = timeout
        self.last_error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def status(self) -> Dict[str, Any]:
        return {
            "provider": "gemini" if self.available else "offline-template",
            "model": self.model if self.available else None,
            "available": self.available,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------ low level
    def _call_gemini(self, model: str, user_text: str, json_output: bool) -> str:
        body: Dict[str, Any] = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.4},
        }
        if json_output:
            body["generationConfig"]["responseMimeType"] = "application/json"
        req = urllib.request.Request(
            GEMINI_ENDPOINT.format(model=model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    def _generate(self, user_text: str, json_output: bool) -> Optional[str]:
        if not self.available:
            self.last_error = "GEMINI_API_KEY not set"
            return None
        models: List[str] = [self.preferred_model] + [m for m in FALLBACK_MODELS if m != self.preferred_model]
        for model in models:
            for attempt in range(2):
                try:
                    text = self._call_gemini(model, user_text, json_output)
                    self.model = model
                    self.last_error = ""
                    return text
                except urllib.error.HTTPError as exc:
                    raw = exc.read().decode("utf-8", "ignore")
                    try:
                        err = json.loads(raw).get("error", {})
                        detail = err.get("message") or raw[:200]
                    except json.JSONDecodeError:
                        detail = raw[:200]
                    self.last_error = f"HTTP {exc.code}: {detail}"
                    if exc.code in (500, 503) and attempt == 0:
                        time.sleep(1.5)  # transient overload -> one quick retry
                        continue
                    if exc.code in (404, 429, 500, 503):
                        break  # retired model / quota / overload -> try the next model
                    return None  # auth or request errors won't be fixed by another model
                except TimeoutError as exc:  # model overloaded/slow -> try the next one
                    self.last_error = f"TimeoutError: {exc}"
                    break
                except Exception as exc:  # network down, malformed reply
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    return None
        return None

    # ------------------------------------------------------------------ features
    def session_report(self, context: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
        lang_name = LANGUAGE_NAMES.get(language, "English")
        prompt = (
            "Write a post-session report from this JSON data.\n"
            "Return ONLY a JSON object with keys:\n"
            f'  "patient_summary": 3-5 sentences in {lang_name} for the patient,\n'
            f'  "next_session_tip": one actionable sentence in {lang_name},\n'
            '  "clinician_summary": 3-6 bullet-style sentences in English for the physiotherapist '
            "(ROM, tempo, symmetry, fatigue, consistency, trend vs previous sessions),\n"
            '  "flags": list of short English strings for anything the physio should check (may be empty).\n\n'
            f"DATA:\n{json.dumps(context, ensure_ascii=False, indent=1)}"
        )
        raw = self._generate(prompt, json_output=True)
        if raw:
            try:
                parsed = json.loads(raw)
                return {
                    "patient_summary": str(parsed.get("patient_summary", "")),
                    "next_session_tip": str(parsed.get("next_session_tip", "")),
                    "clinician_summary": parsed.get("clinician_summary", ""),
                    "flags": parsed.get("flags", []) or [],
                    "language": language,
                    "source": f"gemini:{self.model}",
                }
            except json.JSONDecodeError:
                self.last_error = "Model returned non-JSON output"
        report = offline_report(context, language)
        report["ai_error"] = self.last_error
        return report

    def answer_question(self, question: str, context: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
        lang_name = LANGUAGE_NAMES.get(language, "English")
        prompt = (
            f"A physiotherapist asks about their patient: {question!r}\n"
            f"Answer in {lang_name} in under 150 words, using only this patient data. "
            "If the data cannot answer it, say so.\n\n"
            f"DATA:\n{json.dumps(context, ensure_ascii=False, indent=1)}"
        )
        raw = self._generate(prompt, json_output=False)
        if raw:
            return {"answer": raw, "source": f"gemini:{self.model}"}
        return {
            "answer": "AI assistant unavailable (" + (self.last_error or "no API key") + "). "
            "Set GEMINI_API_KEY in .env to enable questions.",
            "source": "offline",
        }


# ---------------------------------------------------------------------- offline fallback
_TEMPLATES = {
    "en": {
        "summary": "You completed {reps} repetitions of {exercise} in {dur:.0f} seconds. "
        "Your best angle was {peak}° and your movement consistency was {score:.0f}%. {trend}",
        "trend_up": "That is better than your last session — well done!",
        "trend_down": "Slightly lower than last time; rest well and keep practising.",
        "trend_same": "That matches your last session — keep it steady!",
        "trend_none": "Keep practising regularly to see progress.",
        "tip_fast": "Next time, move more slowly and with control.",
        "tip_ok": "Next time, try to keep the same smooth pace for every repetition.",
    },
    "hi": {
        "summary": "आपने {dur:.0f} सेकंड में {exercise} के {reps} दोहराव पूरे किए। "
        "आपका सबसे अच्छा कोण {peak}° रहा और गति की स्थिरता {score:.0f}% रही। {trend}",
        "trend_up": "यह पिछले सत्र से बेहतर है — बहुत बढ़िया!",
        "trend_down": "पिछली बार से थोड़ा कम है; अच्छे से आराम करें और अभ्यास जारी रखें।",
        "trend_same": "यह पिछले सत्र जितना ही है — ऐसे ही जारी रखें!",
        "trend_none": "प्रगति देखने के लिए नियमित अभ्यास करते रहें।",
        "tip_fast": "अगली बार धीरे और नियंत्रण के साथ व्यायाम करें।",
        "tip_ok": "अगली बार हर दोहराव में एक जैसी धीमी गति रखने की कोशिश करें।",
    },
    "mr": {
        "summary": "तुम्ही {dur:.0f} सेकंदात {exercise} च्या {reps} पुनरावृत्ती पूर्ण केल्या. "
        "तुमचा सर्वोत्तम कोन {peak}° होता आणि हालचालीची स्थिरता {score:.0f}% होती. {trend}",
        "trend_up": "हे मागील सत्रापेक्षा चांगले आहे — खूप छान!",
        "trend_down": "मागच्या वेळेपेक्षा थोडे कमी आहे; नीट विश्रांती घ्या आणि सराव सुरू ठेवा.",
        "trend_same": "हे मागच्या सत्राइतकेच आहे — असेच सुरू ठेवा!",
        "trend_none": "प्रगतीसाठी नियमित सराव करत राहा.",
        "tip_fast": "पुढच्या वेळी हळू आणि नियंत्रणाने व्यायाम करा.",
        "tip_ok": "पुढच्या वेळी प्रत्येक पुनरावृत्तीत सारखाच हळू वेग ठेवण्याचा प्रयत्न करा.",
    },
}


def offline_report(context: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    """Deterministic multilingual report used when the Gemini API is unavailable."""
    t = _TEMPLATES.get(language, _TEMPLATES["en"])
    s = context.get("session", {})
    reps = context.get("rep_analysis", {}) or {}
    history = context.get("previous_sessions", []) or []
    peak = reps.get("best_peak_angle") or s.get("max_angle") or "--"

    trend = t["trend_none"]
    if history and history[-1].get("best_peak_angle") is not None and isinstance(peak, (int, float)):
        direction = context.get("exercise", {}).get("direction", "extend")
        prev = history[-1]["best_peak_angle"]
        if abs(peak - prev) < 2.0:
            trend = t["trend_same"]
        else:
            better = peak > prev if direction == "extend" else peak < prev
            trend = t["trend_up"] if better else t["trend_down"]

    too_fast = (reps.get("max_velocity") or 0) > context.get("exercise", {}).get("max_velocity_limit", 1e9)

    flags: List[str] = []
    if reps.get("symmetry_gap_deg") and reps["symmetry_gap_deg"] > 15:
        flags.append(f"Left/right peak gap {reps['symmetry_gap_deg']}°")
    if reps.get("fatigue_rom_drop_pct") and reps["fatigue_rom_drop_pct"] > 20:
        flags.append(f"ROM dropped {reps['fatigue_rom_drop_pct']}% from first to last reps (fatigue?)")
    if too_fast:
        flags.append(f"Peak angular velocity {reps.get('max_velocity')}°/s exceeded limit")

    clinician = (
        f"Reps {s.get('total_reps')}, duration {s.get('duration_seconds')}s, "
        f"angle range {s.get('min_angle')}°–{s.get('max_angle')}°, consistency {s.get('consistency_score')}%. "
        f"Avg ROM/rep {reps.get('avg_rom_deg')}°, avg rep time {reps.get('avg_rep_duration_s')}s."
        + (
            f" ROM variability {reps['rom_variability_pct']}%."
            if reps.get("rom_variability_pct") is not None
            else ""
        )
    )
    return {
        "patient_summary": t["summary"].format(
            reps=s.get("total_reps", 0),
            exercise=s.get("exercise_name", ""),
            dur=float(s.get("duration_seconds") or 0),
            peak=peak,
            score=float(s.get("consistency_score") or 0),
            trend=trend,
        ),
        "next_session_tip": t["tip_fast"] if too_fast else t["tip_ok"],
        "clinician_summary": clinician,
        "flags": flags,
        "language": language,
        "source": "offline-template",
    }
