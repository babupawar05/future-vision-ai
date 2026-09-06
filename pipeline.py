import io
import os
import re
import time
import uuid
import json
import traceback
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError

API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6Kxz-Ts6nWnDDrvwRbU8H40x47NO96XzfNuHX7SXjDQxw")
client = genai.Client(api_key=API_KEY)

SESSION_STORE: dict[str, dict] = {}

MODEL_CASCADE = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

def _extract_retry_delay(err_str: str) -> float:
    match = re.search(r"retry(?:\s*in|Delay)?[:\s]+(\d+(?:\.\d+)?)s?", err_str, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, 45.0)
    return 10.0

def _execute_with_cascade(contents, config, max_retries_per_model: int = 2) -> str:
    last_exception = None

    for model_name in MODEL_CASCADE:
        for attempt in range(max_retries_per_model):
            try:
                print(f"[PIPELINE] Querying {model_name} (Attempt {attempt + 1}/{max_retries_per_model})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response.text

            except APIError as e:
                last_exception = e
                err_text = str(e)

                if e.code == 404 or "NOT_FOUND" in err_text:
                    print(f"[PIPELINE] Model {model_name} not found. Skipping to next candidate...")
                    break
                elif e.code == 429 or "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                    if "GenerateRequestsPerDay" in err_text:
                        print(f"[PIPELINE] Daily limit reached for {model_name}. Cascading...")
                        break
                    wait_sec = _extract_retry_delay(err_text)
                    print(f"[PIPELINE] Throttled on {model_name}. Sleeping {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                elif e.code == 503 or "503" in err_text or "UNAVAILABLE" in err_text:
                    wait_sec = (2 ** attempt) + 1
                    print(f"[PIPELINE] High demand (503) on {model_name}. Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise e
            except Exception as e:
                last_exception = e
                raise e

    if last_exception:
        raise last_exception
    raise RuntimeError("All model options exhausted.")


def generate_palm_reading(image_bytes: bytes, language: str = "English") -> tuple[str, str, dict]:
    """Processes palm image, returns reading text, session_id, and dynamic telemetry scores."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        max_dim = 1400
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        prompt = f"""
        Act as an authoritative, master Vedic palmistry scholar (Hast Rekha Shastra expert) for "Future Vision AI".
        Inspect the contours, epidermal line depths, mounts, and structural markings on this palm image.

        CRITICAL INSTRUCTIONS:
        1. Write the consultation strictly in: **{language}**.
        2. Use traditional terminology (Life Line / Jeevan Rekha, Head Line / Mastishk Rekha, Heart Line / Hriday Rekha, Fate Line / Bhagya Rekha, Mounts of Jupiter, Venus, Saturn, Mercury).
        3. Maintain an objective, empowering, and deeply insightful tone.

        Structure the main consultation using these exact headers:
        ### Hand Archetype & Temperament
        ### Life Line, Vitality & Health
        ### Head Line & Intellectual Capacity
        ### Heart Line & Relationships
        ### Fate Line, Career & Profession
        ### Key Mounts & Auspicious Signs
        ### Wealth Trajectory & Practical Guidance

        AT THE VERY END OF YOUR RESPONSE, provide a raw JSON telemetry block enclosed exactly within `<!-- BIO_TELEMETRY:` and `-->` with integer scores between 40 and 99 and short status words based on actual palm features:
        <!-- BIO_TELEMETRY:
        {{
          "vitality": {{"score": 88, "status": "Robust Vitality"}},
          "career": {{"score": 84, "status": "Strong Momentum"}},
          "love": {{"score": 91, "status": "Deep Harmony"}},
          "intellect": {{"score": 86, "status": "Strategic Clarity"}},
          "wealth": {{"score": 79, "status": "Ascending Peak"}}
        }}
        -->
        """

        config = types.GenerateContentConfig(temperature=0.6)
        raw_output = _execute_with_cascade(contents=[img, prompt], config=config)

        # Fallback values if block extraction fails
        telemetry = {
            "vitality": {"score": 85, "status": "Optimal Vitality"},
            "career": {"score": 82, "status": "High Alignment"},
            "love": {"score": 88, "status": "Harmonious"},
            "intellect": {"score": 86, "status": "Sharp Analytical"},
            "wealth": {"score": 80, "status": "Expanding Cycle"}
        }

        telemetry_match = re.search(r"<!--\s*BIO_TELEMETRY:\s*(\{.*?\})\s*-->", raw_output, re.DOTALL)
        clean_report = raw_output
        if telemetry_match:
            try:
                parsed = json.loads(telemetry_match.group(1))
                telemetry.update(parsed)
                clean_report = raw_output[:telemetry_match.start()].strip()
            except Exception:
                pass

        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "initial_report": clean_report,
            "language": language
        }

        return clean_report, session_id, telemetry

    except Exception as e:
        traceback.print_exc()
        raise e


def ask_palm_question(session_id: str, question: str, language: str = "English") -> str:
    """Delivers structured follow-up consultation referencing the initial reading."""
    if session_id not in SESSION_STORE:
        raise ValueError("Session expired or not found. Please analyze a palm first.")

    session_data = SESSION_STORE[session_id]
    initial_report = session_data["initial_report"]

    prompt = f"""
    You are an elite Vedic palmistry master and predictive analyst for "Future Vision AI".
    The client is asking a specific follow-up inquiry regarding their palm analysis.

    INITIAL PALM ANALYSIS DATA:
    \"\"\"
    {initial_report}
    \"\"\"

    CLIENT INQUIRY:
    "{question}"

    CRITICAL INSTRUCTIONS:
    1. Respond strictly in: **{language}**.
    2. Deliver an authoritative, structured, and deep reading.
    3. Structure your response using these exact markdown headers:
       ### [CORE VERDICT]
       (A direct, impactful 2-sentence executive summary answering the question directly.)
       
       ### [LINE & MOUNT EVIDENCE]
       (Cite specific markers from the hand: line curvature, branchings, mounts, elevation, depth, or auspicious signs that prove this verdict.)
       
       ### [TIMELINE & HORIZON FORECAST]
       (Provide projected age phases, timing windows, or momentum cycles indicated by the lines.)
       
       ### [STRATEGIC GUIDANCE & REMEDY]
       (Empowering, practical Vedic action steps, mindset alignments, or gemstone/planetary remedies.)
    """

    config = types.GenerateContentConfig(temperature=0.65)
    answer_text = _execute_with_cascade(contents=prompt, config=config)
    return answer_text