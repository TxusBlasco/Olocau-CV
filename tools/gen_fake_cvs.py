#!/usr/bin/env python3
"""
Generate realistic CV PDFs with:
- Text via OpenRouter LLM (requires OPENROUTER_API_KEY)
- Photorealistic headshots via Stability Core Images API (requires STABILITY_API_KEY)
- Output defaults to project_root/data/samples/fake_cvs (sibling of tools/)

Example:
  export OPENROUTER_API_KEY="sk-or-..."
  export STABILITY_API_KEY="sk-stability-..."
  python tools/gen_fake_cvs.py --n 30 --llm-model mistralai/mistral-7b-instruct:free
"""

import os
import io
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import requests
from PIL import Image
from openai import OpenAI  # OpenRouter-compatible

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
)

# --------------------------- Helpers -----------------------------------------

def require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing environment variable: {var}")
    return val

def slugify(s: str) -> str:
    s = s.lower().strip().replace(" ", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch == "-")

def tlog(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --------------------------- LLM (OpenRouter) --------------------------------

OPENROUTER_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

def openrouter_client() -> OpenAI:
    key = require_env("OPENROUTER_API_KEY")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)

def gen_cv_json(client: OpenAI, model: str, seed: int, llm_timeout: float, llm_max_retries: int = 2) -> Dict[str, Any]:
    """
    Ask the LLM for a full CV JSON. The LLM decides:
      name (country-consistent), city+country (real), role (varied), summary, skills,
      experiences, education, languages, links.
    Retries lightly on rate-limit.
    """
    sys = (
        "You generate realistic, concise CV content for professional roles. "
        "Return STRICT JSON only (no extra text). "
        "Ensure: (1) full name common in the generated country, "
        "(2) city-country is a real, geographically valid pair, "
        "(3) assign a realistic role (tech/business/engineering/etc.), varied across CVs, "
        "(4) dates in 'Mon YYYY' format, "
        "(5) tone: professional, concise, factual."
    )
    user = {
        "task": "compose_full_cv",
        "seed": seed,
        "constraints": {
            "tone": "professional, concise, factual",
            "dates_format": "Mon YYYY",
            "bullets_per_experience": [3, 5],
            "name_constraints": "Choose a realistic first name and surname typical of the generated country.",
            "location_constraints": "Generate a valid city and country pair that exists in reality.",
            "role_constraints": "Assign a realistic professional role, varying across CVs."
        },
        "return_format": {
            "name": "string", "email": "string", "phone": "string",
            "city": "string", "country": "string", "role": "string",
            "summary": "string(2-3 sentences)",
            "technical_skills": ["string", "..."],
            "soft_skills": ["string", "..."],
            "experiences": [{
                "title": "string", "company": "string",
                "start": "Mon YYYY", "end": "Mon YYYY or Present",
                "bullets": ["string", "..."]
            }],
            "education": [{
                "degree": "string", "institution": "string", "grad_year": "int"
            }],
            "languages": {"Language": "A2/B1/B2/C1/C2"},
            "links": {"LinkedIn": "url", "GitHub": "url"}
        }
    }

    # light retry on rate limits
    backoff = 3.0
    for attempt in range(1, llm_max_retries + 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": json.dumps(user)}],
                response_format={"type": "json_object"},
                timeout=llm_timeout,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            required = ["name","email","phone","city","country","role","summary","experiences","education"]
            for key in required:
                if key not in data:
                    raise ValueError(f"LLM JSON missing key: {key}")
            if not data["name"].strip() or not data["city"].strip() or not data["country"].strip():
                raise ValueError("Invalid empty name/city/country returned by LLM.")
            return data
        except Exception as e:
            msg = str(e)
            if "Rate limit exceeded" in msg or "429" in msg:
                if attempt <= llm_max_retries:
                    tlog(f"LLM rate-limited, retry {attempt}/{llm_max_retries} in {backoff*attempt:.1f}s …")
                    time.sleep(backoff * attempt)
                    continue
            raise

# ---------------------- Stability.ai Photorealistic Headshot ------------------

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))

def stability_core_headshot(name: str, role: str, seed: int, out_path: Path,
                            img_timeout: float = 90.0, max_retries: int = 3, backoff: float = 3.0):
    """
    Stable Image Core (v2beta) — photorealistic, fast.
    IMPORTANT: 'Accept: image/*' must be set, or the API returns JSON.
    """
    api_key = require_env("STABILITY_API_KEY")
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    prompt = (
        f"Professional studio headshot portrait of {name}, a {role}. "
        "Neutral background, natural lighting, sharp focus, shoulders and head visible, "
        "no text, no watermark, photorealistic."
    )
    files = {
        "prompt": (None, prompt),
        "aspect_ratio": (None, "1:1"),
        "output_format": (None, "png"),
        "seed": (None, str(seed)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        r = requests.post(url, headers=headers, files=files, timeout=img_timeout)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image/"):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = center_crop_square(img).resize((512, 512))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="PNG")
            return
        last_err = f"{r.status_code} {r.text[:200]}"
        time.sleep(backoff * attempt)
    raise RuntimeError(f"Stability API error after {max_retries} attempts: {last_err}")

# -------------------------- PDF Rendering ------------------------------------

def render_pdf(path: Path, cv: Dict[str, Any], photo_path: Path):
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Header", fontSize=18, leading=22, spaceAfter=6, textColor=colors.HexColor("#222")))
    styles.add(ParagraphStyle(name="SubHeader", fontSize=11, leading=14, textColor=colors.HexColor("#444")))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.HexColor("#333")))
    styles.add(ParagraphStyle(name="Section", fontSize=12, leading=14, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#111")))

    flow = []
    left_flow = [
        Paragraph(cv["name"], styles["Header"]),
        Paragraph(f'{cv["city"]}, {cv["country"]}  •  {cv["email"]}  •  {cv["phone"]}', styles["SubHeader"])
    ]
    if isinstance(cv.get("links"), dict) and cv["links"]:
        links_line = "  •  ".join([f"{k}: {v}" for k, v in cv["links"].items()])
        left_flow.append(Paragraph(links_line, styles["Small"]))
    left_flow.append(Spacer(1, 6))
    left_table = Table([[cell] for cell in left_flow], colWidths=[120*mm])
    left_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))

    rl_img = RLImage(str(photo_path), width=40*mm, height=40*mm)
    header_table = Table([[left_table, rl_img]], colWidths=[140*mm, 40*mm])
    header_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    flow += [header_table, HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#999")), Spacer(1, 6)]
    flow += [Paragraph(cv.get("role",""), styles["Section"]),
             Paragraph(cv.get("summary",""), styles["Body"])]

    if "technical_skills" in cv:
        flow += [Paragraph("Technical Skills", styles["Section"]),
                 Paragraph(", ".join(cv["technical_skills"]), styles["Body"])]
    if "soft_skills" in cv:
        flow += [Paragraph("Soft Skills", styles["Section"]),
                 Paragraph(", ".join(cv["soft_skills"]), styles["Body"])]

    flow.append(Paragraph("Experience", styles["Section"]))
    for exp in cv["experiences"]:
        date_fmt = f'{exp.get("start","")} – {exp.get("end","")}'
        flow.append(Paragraph(f'<b>{exp.get("title","")}</b> — {exp.get("company","")}  ({date_fmt})', styles["Body"]))
        bullets = [[f'• {b}'] for b in exp.get("bullets", [])]
        t = Table(bullets, colWidths=[170*mm])
        t.setStyle(TableStyle([
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 1),
            ("BOTTOMPADDING", (0,0), (-1,-1), 1),
        ]))
        flow += [t, Spacer(1, 2)]

    flow.append(Paragraph("Education", styles["Section"]))
    for edu in cv["education"]:
        line = f'{edu.get("degree","")} — {edu.get("institution","")} ({edu.get("grad_year","")})'
        flow.append(Paragraph(line, styles["Body"]))

    if isinstance(cv.get("languages"), dict) and cv["languages"]:
        flow += [Paragraph("Languages", styles["Section"]),
                 Paragraph(", ".join([f"{k} ({v})" for k, v in cv["languages"].items()]), styles["Body"])]

    doc.build(flow)

# --------------------------- Main --------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=28, help="Number of CVs")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--sleep", type=float, default=0.6, help="Delay between LLM requests (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for Stability photos")
    parser.add_argument("--llm-model", type=str, default="mistralai/mistral-7b-instruct:free",
                        help="OpenRouter model id, e.g., mistralai/mistral-7b-instruct:free, "
                             "meta-llama/llama-3.1-8b-instruct, google/gemma-7b-it:free, etc.")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="Timeout per LLM request (seconds)")
    # Optional override, default resolves to project_root/data/samples/fake_cvs
    project_root = Path(__file__).resolve().parent.parent
    default_outdir = project_root / "data" / "samples" / "fake_cvs"
    parser.add_argument("--outdir", type=str, default=str(default_outdir),
                        help="Output folder (defaults to project_root/data/samples/fake_cvs)")
    args = parser.parse_args()

    _ = require_env("OPENROUTER_API_KEY")
    _ = require_env("STABILITY_API_KEY")

    outdir = Path(args.outdir).resolve()
    photos_dir = outdir / "photos"
    outdir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    client = openrouter_client()
    model = args.llm_model

    index_path = outdir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["name","email","phone","role","city","country","file","photo"])

    seen_names = set()

    tlog(f"LLM model: {model}")
    for i in range(args.n):
        ts = time.monotonic()
        cv_json = gen_cv_json(client=client, model=model, seed=args.seed + i,
                              llm_timeout=args.llm_timeout, llm_max_retries=2)
        # ensure unique names
        tries = 0
        while cv_json["name"] in seen_names and tries < 3:
            cv_json = gen_cv_json(client=client, model=model, seed=args.seed + 100 + i + tries,
                                  llm_timeout=args.llm_timeout, llm_max_retries=2)
            tries += 1
        if cv_json["name"] in seen_names:
            cv_json["name"] = f'{cv_json["name"]} {i+1}'
        seen_names.add(cv_json["name"])
        tlog(f"text {i+1}/{args.n} OK ({time.monotonic()-ts:.1f}s) — {cv_json['name']}")

        # photo
        slug = slugify(cv_json["name"])
        photo_path = photos_dir / f"{slug}_{i+1:02d}.png"
        ts = time.monotonic()
        stability_core_headshot(
            name=cv_json["name"],
            role=cv_json.get("role","professional"),
            seed=args.seed + i,
            out_path=photo_path,
            img_timeout=90.0,
            max_retries=args.max_retries
        )
        tlog(f"photo {i+1}/{args.n} OK ({time.monotonic()-ts:.1f}s) → {photo_path.name}")

        # pdf
        pdf_path = outdir / f"cv_{slug}_{i+1:02d}.pdf"
        render_pdf(pdf_path, cv_json, photo_path)
        tlog(f"pdf   {i+1}/{args.n} OK → {pdf_path.name}")

        with index_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                cv_json["name"], cv_json["email"], cv_json["phone"],
                cv_json["role"], cv_json["city"], cv_json["country"],
                pdf_path.name, photo_path.name
            ])

        time.sleep(args.sleep)

    tlog(f"DONE → {outdir}")
    tlog(f"Index → {index_path}")

if __name__ == "__main__":
    main()
