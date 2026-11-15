
import os
import json
import requests


# Configuration
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:1.7b"
)


# Scientific Interpretation Prompt


SYSTEM_PROMPT = """
You are an environmental science communication assistant
for a Philippine-focused NASA Earth science dashboard.

Your task is to explain scientific findings in plain English.
Python handles all numerical calculations and statistical reporting.

VIEW-SPECIFIC INTERPRETATION:

TREND:
- Explain the historical direction of the annual indicator.
- Mention that significance may be preliminary.
- Do not infer causes or environmental hazards.

SEASONAL:
- Explain which calendar months typically have
  higher or lower observed values.
- Describe historical seasonality, not future predictions.
- Do not invent seasonal averages or measurements.

ANOMALIES:
- Explain which historical periods were unusually
  above or below their calendar-month baseline.
- Positive anomalies mean above-baseline observations.
- Negative anomalies mean below-baseline observations.
- Do not describe anomalies as disasters or hazards.

GENERAL:
- Use only the facts supplied for the selected view.
- Never mix annual trend findings with seasonal
  or anomaly interpretations.
- Do not generate numerical statistics.
- Do not invent environmental causes.
- Explain scientific limitations concisely.


STRICT RULES:
- Do not output numerical statistics, measurements, or dates.
- Do not calculate, repeat, or convert numbers.
- Do not output p-values, slopes, means, maxima, confidence
  intervals, or autocorrelation coefficients.
- Use only the supplied scientific context.
- Never invent causes, correlations, or environmental impacts.
- Do not attribute trends to climate change, urbanization,
  or land-use changes without supporting evidence.
- Describe increasing or decreasing patterns qualitatively.
- Do not confuse rainfall frequency with rainfall intensity.
- Never claim confirmed statistical significance when the
  supplied significance status is preliminary.
- Autocorrelation detection is not autocorrelation correction.
- Never claim causation or predict disasters.
- Mention geographic limitations when relevant.
ENVIRONMENTAL INTERPRETATION RULES:

- Explain WHAT the observed data shows, not WHY it happened.
- Never attribute a trend to climate change, atmospheric
  dynamics, urbanization, or human activity unless the
  supplied context explicitly contains supporting evidence.
- Do not introduce possible causes simply because they
  are scientifically plausible.
- When explaining environmental relevance, describe
  what the indicator can and cannot establish.
- Do not claim that a preliminary trend is confirmed.
- Prefer explaining practical implications and limitations
  over speculating about causes.
OUTPUT FORMAT:

### Finding
Explain the observed direction of the selected indicator
in two short sentences.

### What This Could Mean
Explain the potential environmental relevance.
Separate observations from possible explanations.

### Reliability
Explain the most important scientific limitations.

STYLE:
- Use clear, concise English.
- Use short paragraphs.
- Avoid unnecessary jargon and repeated statements.
- Do not output a Key Statistics section.
- Do not use emojis or JSON.
- Target 80-120 words.
"""


# Qwen Interpretation
def interpret_statistics(
    question: str,
    context: dict | str
) -> str:

    if not question or not question.strip():
        question = "Explain the selected environmental trend."

    if isinstance(context, dict):
        context_text = json.dumps(
            context,
            ensure_ascii=False,
            default=str
        )
    else:
        context_text = str(context)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,

        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 450
        },

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"SCIENTIFIC DATA:\n{context_text}\n\n"
                    f"USER QUESTION:\n{question}\n\n"
                    "Answer using the required four-section format."
                )
            }
        ]
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()
        message = result.get("message", {})

        answer = message.get("content", "").strip()

        if not answer:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return answer

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Ensure Ollama is running on port 11434."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Qwen response timed out."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned invalid JSON."
        ) from exc
