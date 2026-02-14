import os
import json
import csv
import time
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
import anthropic

import sqlite3
from datetime import datetime

# ------------------- Config -------------------
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
TAXONOMY_VERSION = os.getenv("TAXONOMY_VERSION", "v1")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
MAX_TOKENS = 600
TEMPERATURE = 0

# Weighted trust in each field (tune later)
FIELD_WEIGHTS = {
    "role_title": 1/3,
    "job_title": 1/3,
    "vendor_role": 1/3,
}

# Candidate filtering + decision thresholds
MIN_CANDIDATE_CONF = 0.10   # ignore low-confidence candidates per field
MIN_TOTAL_SCORE = 0.70      # if top aggregated score below this -> review
MIN_MARGIN = 0.10           # if top-2 too close -> review

OUTPUT_CSV = "role_taxonomy_ensemble_results.csv"

# Canonical roles (v1: ~30)
CANONICAL_ROLES = [
    "Backend Engineer",
    "Frontend Engineer",
    "Full Stack Engineer",
    "Mobile Engineer",
    "Platform Engineer",
    "DevOps Engineer",
    "Embedded Software Engineer",

    "QA Engineer (Manual)",
    "QA Engineer (Automation)",
    "SDET (Software Development Engineer in Test)",
    "Performance Test Engineer",

    "Data Engineer",
    "Analytics Engineer",
    "BI Engineer",
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "MLOps Engineer",

    "Site Reliability Engineer (SRE)",
    "Cloud Infrastructure Engineer",
    "Systems Engineer",
    "Systems Administrator",
    "Network Engineer",

    "Security Engineer",
    "Security Analyst",

    "Technical Program Manager (TPM)",
    "Program Manager",
    "Product Manager",
    "Solutions Engineer",
]

# Family mapping (deterministic, not inferred)
CANONICAL_TO_FAMILY = {
    # Software engineering
    "Backend Engineer": "Software Engineering",
    "Frontend Engineer": "Software Engineering",
    "Full Stack Engineer": "Software Engineering",
    "Mobile Engineer": "Software Engineering",
    "Platform Engineer": "Software Engineering",
    "DevOps Engineer": "Software Engineering",
    "Embedded Software Engineer": "Software Engineering",

    # Quality
    "QA Engineer (Manual)": "Quality Engineering",
    "QA Engineer (Automation)": "Quality Engineering",
    "SDET (Software Development Engineer in Test)": "Quality Engineering",
    "Performance Test Engineer": "Quality Engineering",

    # Data/ML
    "Data Engineer": "Data / ML",
    "Analytics Engineer": "Data / ML",
    "BI Engineer": "Data / ML",
    "Data Analyst": "Data / ML",
    "Data Scientist": "Data / ML",
    "Machine Learning Engineer": "Data / ML",
    "MLOps Engineer": "Data / ML",

    # Infra / systems
    "Site Reliability Engineer (SRE)": "Infrastructure / Systems",
    "Cloud Infrastructure Engineer": "Infrastructure / Systems",
    "Systems Engineer": "Infrastructure / Systems",
    "Systems Administrator": "Infrastructure / Systems",
    "Network Engineer": "Infrastructure / Systems",

    # Security
    "Security Engineer": "Security",
    "Security Analyst": "Security",

    # Program/product/solutions
    "Technical Program Manager (TPM)": "Program / Product",
    "Program Manager": "Program / Product",
    "Product Manager": "Program / Product",
    "Solutions Engineer": "Program / Product",
}

# Import CSV datasource
INPUT_CSV = "input_records.csv"

# ------------------- Helpers -------------------
def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def parse_json_safely(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = strip_code_fences(raw_text)
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e} | raw={raw_text!r}"


def build_field_prompt(field_name: str, field_text: str) -> str:
    """
    Returns prompt asking for top-3 canonical roles for a specific field.
    Also extracts optional specialization + level when explicit.
    """
    return (
        "You are a strict taxonomy classifier.\n"
        f"Allowed canonical roles: {CANONICAL_ROLES}\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "field": "<field_name>",\n'
        '  "text": "<original_text>",\n'
        '  "candidates": [\n'
        '    {"canonical_role": "<one of allowed canonical roles>", "confidence": <0..1>}\n'
        "  ],\n"
        '  "level": <1..5 or null>,\n'
        '  "specialization": "<string or null>"\n'
        "}\n"
        "Rules:\n"
        "- Always return exactly 3 candidates, sorted by confidence desc.\n"
        "- Each confidence must be between 0 and 1.\n"
        "- The sum of confidences across the 3 candidates must equal 1.0.\n"
        "- Avoid ties unless truly indistinguishable.\n"
        "- Do not output roles outside the allowed list.\n"
        "- level must be null unless the text explicitly includes a seniority keyword.\n"
        "- Seniority keywords: Junior=1, Mid=3, Senior=4, Lead/Principal/Staff=5.\n"
        "- specialization must be null unless explicitly stated (e.g., Android, iOS, Backend, Frontend, Automation, Kubernetes).\n"
        "- Output JSON only. No markdown.\n"
        "- If the input text is generic (e.g., 'Software Engineer'), prefer broader canonical roles over very specific ones.\n"
        f'Now classify field="{field_name}" text="{field_text}".'
    )

def normalize_term(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

CACHE_DB = "cache.sqlite"

def cache_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS term_cache (
            norm_term TEXT NOT NULL,
            model TEXT NOT NULL,
            taxonomy_version TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (norm_term, model, taxonomy_version, prompt_version)
        )
    """)
    return conn

def cache_get(conn: sqlite3.Connection, term: str) -> Optional[Dict[str, Any]]:
    norm = normalize_term(term)
    cur = conn.execute("""
        SELECT result_json
        FROM term_cache
        WHERE norm_term = ? AND model = ? AND taxonomy_version = ? AND prompt_version = ?
    """, (norm, MODEL, TAXONOMY_VERSION, PROMPT_VERSION))
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row[0])

def cache_set(conn: sqlite3.Connection, term: str, result: Dict[str, Any]) -> None:
    norm = normalize_term(term)
    conn.execute("""
        INSERT OR REPLACE INTO term_cache
        (norm_term, model, taxonomy_version, prompt_version, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        norm,
        MODEL,
        TAXONOMY_VERSION,
        PROMPT_VERSION,
        json.dumps(result),
        datetime.utcnow().isoformat()
    ))
    conn.commit()

def classify_field(
    conn: sqlite3.Connection,
    client: anthropic.Anthropic,
    field_name: str,
    field_text: str
) -> Dict[str, Any]:

    # 1) Cache lookup first
    cached = cache_get(conn, field_text)
    if cached is not None:
        return {
            "field": field_name,
            "text": field_text,
            "candidates": cached.get("candidates", []),
            "level": cached.get("level"),
            "specialization": cached.get("specialization"),
            "error": cached.get("error"),
        }

    # 2) Call API (cache miss)
    prompt = build_field_prompt(field_name, field_text)

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text
    data, err = parse_json_safely(raw)

    result: Dict[str, Any] = {
        "field": field_name,
        "text": field_text,
        "candidates": [],
        "level": None,
        "specialization": None,
        "error": None,
    }

    if err:
        result["error"] = err
        payload = {
            "candidates": [],
            "level": None,
            "specialization": None,
            "error": err,
        }
        cache_set(conn, field_text, payload)
        return result

    # Parse candidates
    cands = data.get("candidates", [])
    cleaned: List[Dict[str, Any]] = []
    if isinstance(cands, list):
        for c in cands:
            if not isinstance(c, dict):
                continue
            role = c.get("canonical_role")
            conf = c.get("confidence")
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                continue
            if role in CANONICAL_ROLES and conf_f >= MIN_CANDIDATE_CONF:
                cleaned.append({"canonical_role": role, "confidence": conf_f})
    cleaned.sort(key=lambda x: x["confidence"], reverse=True)
    result["candidates"] = cleaned[:3]

    # Parse level & specialization
    lvl = data.get("level")
    spec = data.get("specialization")
    if isinstance(lvl, int) and (1 <= lvl <= 5):
        result["level"] = lvl
    if isinstance(spec, str) and spec.strip():
        result["specialization"] = spec.strip()

    payload = {
        "candidates": result["candidates"],
        "level": result["level"],
        "specialization": result["specialization"],
        "error": None,
    }
    cache_set(conn, field_text, payload)

    return result

def aggregate_candidates(per_field: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Family-first aggregation:
    1) Score families by summing weighted candidate confidences mapped to family.
    2) Pick winning family (with margin).
    3) Within winning family, pick best canonical role by weighted score.
    4) Provide separate review flags for family vs role.

    Assumes:
      - per_field[field]["candidates"] is list of {"canonical_role": str, "confidence": float}
      - CANONICAL_TO_FAMILY maps canonical_role -> family
      - FIELD_WEIGHTS exists
      - MIN_TOTAL_SCORE and MIN_MARGIN exist (we'll apply them to family + role)
    """

    contributing_fields = 0

    # Canonical role scores (same as before, but we keep them for later)
    role_scores: Dict[str, float] = {}

    # Family scores (new)
    family_scores: Dict[str, float] = {}

    for field_name, res in per_field.items():
        weight = FIELD_WEIGHTS.get(field_name, 0.0)
        cands = res.get("candidates", [])
        if cands:
            contributing_fields += 1

        for c in cands:
            role = c["canonical_role"]
            conf = c["confidence"]

            # 1) accumulate role score
            role_scores[role] = role_scores.get(role, 0.0) + weight * conf

            # 2) accumulate family score
            fam = CANONICAL_TO_FAMILY.get(role)
            if fam:
                family_scores[fam] = family_scores.get(fam, 0.0) + weight * conf

    # ---- Rank families ----
    ranked_families = sorted(family_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_family = ranked_families[0][0] if ranked_families else None
    top_family_score = ranked_families[0][1] if ranked_families else 0.0
    second_family_score = ranked_families[1][1] if len(ranked_families) > 1 else 0.0
    family_margin = top_family_score - second_family_score

    # ---- Rank roles within the winning family ----
    ranked_roles_within_family: List[tuple] = []
    if top_family:
        for role, score in role_scores.items():
            if CANONICAL_TO_FAMILY.get(role) == top_family:
                ranked_roles_within_family.append((role, score))
        ranked_roles_within_family.sort(key=lambda kv: kv[1], reverse=True)

    top_role = ranked_roles_within_family[0][0] if ranked_roles_within_family else None
    top_role_score = ranked_roles_within_family[0][1] if ranked_roles_within_family else 0.0
    second_role_score = ranked_roles_within_family[1][1] if len(ranked_roles_within_family) > 1 else 0.0
    role_margin = top_role_score - second_role_score

    # ---- Review logic ----
    # Family review: is the family decision trustworthy?
    family_needs_review = False
    if top_family is None:
        family_needs_review = True
    if contributing_fields < 2:
        family_needs_review = True
    if top_family_score < MIN_TOTAL_SCORE:
        family_needs_review = True
    if family_margin < MIN_MARGIN:
        family_needs_review = True

    # Role review: if family is clear, is the role within family clear?
    role_needs_review = False
    if top_role is None:
        role_needs_review = True

    # If the family itself is uncertain, the role is inherently uncertain
    if family_needs_review:
        role_needs_review = True
    else:
        # Only enforce role thresholds if family is trusted
        if top_role_score < (MIN_TOTAL_SCORE * 0.8):  # slightly looser than family
            role_needs_review = True
        if role_margin < (MIN_MARGIN * 0.8):
            role_needs_review = True

    # Backwards-compatible single flag if you want it:
    needs_review = family_needs_review or role_needs_review

    return {
        # Final decisions
        "final_family": top_family,
        "final_canonical_role": top_role,

        # Family diagnostics
        "family_score": round(top_family_score, 4),
        "family_second_score": round(second_family_score, 4),
        "family_margin": round(family_margin, 4),
        "family_needs_review": family_needs_review,
        "family_score_breakdown": ranked_families[:5],

        # Role diagnostics (within selected family)
        "role_score": round(top_role_score, 4),
        "role_second_score": round(second_role_score, 4),
        "role_margin": round(role_margin, 4),
        "role_needs_review": role_needs_review,
        "role_score_breakdown_within_family": ranked_roles_within_family[:5],

        # General
        "contributing_fields": contributing_fields,
        "needs_review": needs_review,
    }

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "username",
        "role_title",
        "job_title",
        "vendor_role",

        "final_family",
        "final_canonical_role",

        "family_score",
        "family_second_score",
        "family_margin",
        "family_needs_review",

        "role_score",
        "role_second_score",
        "role_margin",
        "role_needs_review",

        "contributing_fields",
        "needs_review",

        "final_level",
        "final_specialization",

        "role_title_candidates",
        "job_title_candidates",
        "vendor_role_candidates",

        "errors",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def combine_level_spec(per_field: Dict[str, Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    """
    Simple rule: prefer values from vendor_role, then role_title, then job_title.
    (You can change this policy later.)
    """
    for field in ["vendor_role", "role_title", "job_title"]:
        lvl = per_field[field].get("level")
        spec = per_field[field].get("specialization")
        if lvl is not None or spec is not None:
            return lvl, spec
    return None, None

def load_records_from_csv(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "username": (row.get("username") or "").strip(),
                "role_title": (row.get("role_title") or "").strip(),
                "job_title": (row.get("job_title") or "").strip(),
                "vendor_role": (row.get("vendor_role") or "").strip(),
            })
    return records

# ------------------- Main -------------------
def main() -> None:
    load_dotenv()
    conn = cache_connect()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env")

    client = anthropic.Anthropic(api_key=api_key)

    out_rows: List[Dict[str, Any]] = []

    records = load_records_from_csv(INPUT_CSV)
    for i, rec in enumerate(records, start=1):
        username = rec["username"]
        print(f"\n[{i}/{len(records)}] username={username}")

        per_field: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        for field_name in ["role_title", "job_title", "vendor_role"]:
            text = (rec.get(field_name) or "").strip()
            res = classify_field(conn, client, field_name, text) if text else {
                "field": field_name, "text": text, "candidates": [], "level": None, "specialization": None, "error": None
            }
            per_field[field_name] = res
            if res.get("error"):
                errors.append(f"{field_name}: {res['error']}")
            time.sleep(0.2)

        agg = aggregate_candidates(per_field)
        final_level, final_spec = combine_level_spec(per_field)

        row = {
            "username": username,
            "role_title": rec.get("role_title"),
            "job_title": rec.get("job_title"),
            "vendor_role": rec.get("vendor_role"),

            # Final decisions
            "final_family": agg["final_family"],
            "final_canonical_role": agg["final_canonical_role"],

            # Family diagnostics (primary)
            "family_score": agg["family_score"],
            "family_second_score": agg["family_second_score"],
            "family_margin": agg["family_margin"],
            "family_needs_review": agg["family_needs_review"],

            # Role diagnostics within chosen family (secondary)
            "role_score": agg["role_score"],
            "role_second_score": agg["role_second_score"],
            "role_margin": agg["role_margin"],
            "role_needs_review": agg["role_needs_review"],

            # General
            "contributing_fields": agg["contributing_fields"],
            "needs_review": agg["needs_review"],

            # Optional extracted attributes
            "final_level": final_level,
            "final_specialization": final_spec,

            # Transparency / auditability
            "role_title_candidates": json.dumps(per_field["role_title"]["candidates"]),
            "job_title_candidates": json.dumps(per_field["job_title"]["candidates"]),
            "vendor_role_candidates": json.dumps(per_field["vendor_role"]["candidates"]),

            # Debug
            "errors": " | ".join(errors) if errors else "",
        }
        out_rows.append(row)

        print("Decision:", agg)

    write_csv(out_rows, OUTPUT_CSV)
    print(f"\nDone. Wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
