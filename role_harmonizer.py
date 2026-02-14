import os
import json
import csv
import time
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
import anthropic

# ---------- Config ----------
MODEL = "claude-haiku-4-5"  # cheap + fast for learning
MAX_TOKENS = 250
TEMPERATURE = 0
CONFIDENCE_THRESHOLD = 0.90  # below this -> needs_review

ALLOWED_ROLE_FAMILIES = [
    "Software Engineering",
    "Quality Engineering",
    "Program Management",
    "Data Engineering",
]

# Titles to test (you can expand this list)
TITLES = [
    "Senior QA Eng - Bangalore - Vendor X",
    "Tester - India",
    "QAT Eng (Automation) - SF Bay Area",
    "Quality Engineer - Seattle",
    "QA Lead - London",
    "SDET - Remote - Vendor Y",
]

OUTPUT_CSV = "role_harmonizer_results.csv"


# ---------- Helpers ----------
def strip_code_fences(text: str) -> str:
    """Remove common markdown code fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # remove starting fence (``` or ```json)
        t = t.split("\n", 1)[-1] if "\n" in t else ""
        # remove ending fence
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def build_prompt(title: str) -> str:
    return (
        "Return ONLY valid JSON with keys: role_family, specialization, level, location, confidence.\n"
        f"Allowed role_family values: {ALLOWED_ROLE_FAMILIES}\n"
        "- specialization must be null unless explicitly stated (e.g., Android, iOS, Backend, Automation).\n"
        "- level must be null unless one of these keywords appears: Junior, Mid, Senior, Lead, Principal, Staff.\n"
        "- Map level keywords: Junior=1, Mid=3, Senior=4, Lead/Principal/Staff=5.\n"
        "- location: extract if present; otherwise null.\n"
        "- confidence must be between 0 and 1.\n"
        f"Map this title: '{title}'"
    )


def parse_json_safely(raw_text: str) -> (Optional[Dict[str, Any]], Optional[str]):
    """Return (data, error)."""
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)} | raw={raw_text!r}"


def classify_title(client: anthropic.Anthropic, title: str) -> Dict[str, Any]:
    prompt = build_prompt(title)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    data, err = parse_json_safely(raw)

    result: Dict[str, Any] = {
        "input_title": title,
        "role_family": None,
        "specialization": None,
        "level": None,
        "location": None,
        "confidence": None,
        "needs_review": True,
        "error": None,
    }

    if err:
        result["error"] = err
        return result

    # Basic field extraction with gentle defaults
    result["role_family"] = data.get("role_family")
    result["specialization"] = data.get("specialization")
    result["level"] = data.get("level")
    result["location"] = data.get("location")
    result["confidence"] = data.get("confidence")

    # Determine needs_review
    try:
        conf = float(result["confidence"]) if result["confidence"] is not None else 0.0
    except (TypeError, ValueError):
        conf = 0.0

    result["needs_review"] = conf < CONFIDENCE_THRESHOLD

    return result


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "input_title",
        "role_family",
        "specialization",
        "level",
        "location",
        "confidence",
        "needs_review",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------- Main ----------
def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env file")

    client = anthropic.Anthropic(api_key=api_key)

    results: List[Dict[str, Any]] = []
    for i, title in enumerate(TITLES, start=1):
        print(f"[{i}/{len(TITLES)}] Classifying: {title}")
        r = classify_title(client, title)
        results.append(r)

        # tiny sleep to be nice to the API (also helps avoid rate limits)
        time.sleep(0.2)

    write_csv(results, OUTPUT_CSV)

    print("\nDone.")
    print(f"Wrote results to: {OUTPUT_CSV}")
    # Print summary
    flagged = sum(1 for r in results if r["needs_review"] or r["error"])
    print(f"Flagged for review (low confidence or error): {flagged}/{len(results)}")


if __name__ == "__main__":
    main()
