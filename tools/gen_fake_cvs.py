#!/usr/bin/env python3
"""
LLM-only CV generator: 25–30 realistic PDFs with AI-generated text + headshots.

Requirements:
  - OPENROUTER_API_KEY (text via OpenRouter/OpenAI SDK)
  - STABILITY_API_KEY (photos via Stability Core Images API)

Outputs:
  - data/samples/fake_cvs/cv_<slug>_<idx>.pdf
  - data/samples/fake_cvs/photos/<slug>_<idx>.jpg
  - data/samples/fake_cvs/index.csv

Usage:
  python tools/gen_fake_cvs.py --n 28 --seed 13
"""

import os
import io
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI           # OpenRouter-compatible
import requests

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage

from PIL import Image

# --------------------------- Configuration -----------------------------------

DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

ROLES = [
    "Data Scientist", "Machine Learning Engineer", "Data Analyst",
    "Backend Engineer", "Full-Stack Engineer", "MLOps Engineer",
    "DevOps Engineer", "Product Manager", "QA Engineer", "Data Engineer",
    "NLP Engineer", "Computer Vision Engineer", "Business Analyst"
]

CITIES = [
    "Valencia", "Barcelona", "Madrid", "Seville", "Bilbao", "Lisbon",
    "Porto", "Milan", "Munich", "Berlin", "Paris", "London"
]
COUNTRIES = [
    "Spain", "Portugal", "Italy", "Germany", "France", "United Kingdom"
]

# ---------------------------- Helpers ----------------------------------------

def require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing environment variable: {var}")
    return val

def slugify(s: str) -> str:
    s = s.lower().replace(" ", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch == "-")

def openrouter_client() -> OpenAI:
    key = require_env("OPENROUTER_API_KEY")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)

def gen_cv_json(
    client: OpenAI,
    model: str,
    seed: int,
    role_hint: str,
    city_hint: str,
    country_hint: str
) -> Dict[str, Any]:
    """
    Ask the LLM for a full CV JSON with:
      - name, email, phone, city, country
      - role (aligned to hint), summary
      - technical_skills[], soft_skills[]
      - experiences[]: {title, company, start, end, bullets[]}
      - education[]: {degree, institution, grad_year}
      - languages{}: {lang: level}
      - links{}: {LinkedIn, GitHub}
    """
    sys = (
        "You generate realistic, concise CV content for technical roles. "
        "Return STRICT JSON, no extra text."
    )
    user = {
        "task": "compose_full_cv",
        "seed": seed,
        "constraints": {
            "tone": "professional, concise, factual",
            "dates_format": "Mon YYYY",
            "bullets_per_experience": [3, 5]
        },
        "hints": {
            "role": role_hint,
            "city": city_hint,
            "country": country_hint
        },
        "return_format": {
            "name": "string",
            "email": "string",
            "phone": "string",
            "city": "string",
            "country": "string",
            "role": "string",
            "summary": "string(2-3 sentences)",
            "technical_skills": ["string", "..."],
            "soft_skills": ["string", "..."],
            "experiences": [
                {
                    "title": "string",
                    "company": "string",
                    "start": "Mon YYYY",
                    "end": "Mon YYYY or Present",
                    "bullets": ["string", "..."]
                }
            ],
            "education": [
                {
                    "degree": "string",
                    "institution": "string",
                    "grad_year": "int"
                }
            ],
            "languages": {"Language": "A2/B1/B2/C1/C2"},
            "links": {"LinkedIn": "url", "GitHub": "url"}
        }
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    # Light sanity checks (hard fail if missing critical fields)
    for key in ["name", "email", "phone", "city", "country", "role", "summary", "experiences", "education"]:
        if key not in data:
            raise ValueError(f"LLM JSON missing key: {key}")
    return data

def stability_headshot(name: str, role: str, seed: int, out_path: Path):
    api_key = require_env("STABILITY_API_KEY")
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    prompt = (
        f"Professional studio headshot portrait of {name}, a {role}. "
        "Neutral background, natural lighting, sharp focus, shoulders and head visible, "
        "no text, photorealistic."
    )
    files = {
        "prompt": (None, prompt),
        "aspect_ratio": (None, "1:1"),
        "output_format": (None, "png"),
        "seed": (None, str(seed)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",  # request an image response
    }
    r = requests.post(url, headers=headers, files=files, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Stability API error: {r.status_code} {r.text[:200]}")
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = center_crop_square(img).resize((512, 512))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))

# -------------------------- PDF Rendering ------------------------------------

def render_pdf(path: Path, cv: Dict[str, Any], photo_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Header", fontSize=18, leading=22, spaceAfter=6, textColor=colors.HexColor("#222")))
    styles.add(ParagraphStyle(name="SubHeader", fontSize=11, leading=14, textColor=colors.HexColor("#444")))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.HexColor("#333")))
    styles.add(ParagraphStyle(name="Section", fontSize=12, leading=14, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#111")))

    flow = []

    # Header with photo
    left_flow = []
    left_flow.append(Paragraph(cv["name"], styles["Header"]))
    contact = f'{cv["city"]}, {cv["country"]}  •  {cv["email"]}  •  {cv["phone"]}'
    left_flow.append(Paragraph(contact, styles["SubHeader"]))
    if "links" in cv and isinstance(cv["links"], dict) and cv["links"]:
        links_line = "  •  ".join([f"{k}: {v}" for k, v in cv["links"].items()])
        left_flow.append(Paragraph(links_line, styles["Small"]))
    left_flow.append(Spacer(1, 6))
    left_table = Table([[cell] for cell in left_flow], colWidths=[120*mm])
    left_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))

    rl_img = RLImage(str(photo_path), width=40*mm, height=40*mm)
    header_table = Table([[left_table, rl_img]], colWidths=[140*mm, 40*mm])
    header_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    flow.append(header_table)

    flow.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#999")))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph(cv.get("role", ""), styles["Section"]))
    flow.append(Paragraph(cv.get("summary", ""), styles["Body"]))

    # Skills
    if "technical_skills" in cv:
        flow.append(Paragraph("Technical Skills", styles["Section"]))
        flow.append(Paragraph(", ".join(cv["technical_skills"]), styles["Body"]))
    if "soft_skills" in cv:
        flow.append(Paragraph("Soft Skills", styles["Section"]))
        flow.append(Paragraph(", ".join(cv["soft_skills"]), styles["Body"]))

    # Experience
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
        flow.append(t)
        flow.append(Spacer(1, 2))

    # Education
    flow.append(Paragraph("Education", styles["Section"]))
    for edu in cv["education"]:
        line = f'{edu.get("degree","")} — {edu.get("institution","")} ({edu.get("grad_year","")})'
        flow.append(Paragraph(line, styles["Body"]))

    # Languages
    if "languages" in cv and isinstance(cv["languages"], dict) and cv["languages"]:
        flow.append(Paragraph("Languages", styles["Section"]))
        flow.append(Paragraph(", ".join([f"{k} ({v})" for k, v in cv["languages"].items()]), styles["Body"]))

    doc.build(flow)

# --------------------------- Main --------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=28, help="Number of CVs (25–30 recommended)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed (controls prompts)")
    parser.add_argument("--outdir", type=str, default="data/samples/fake_cvs", help="Output folder")
    args = parser.parse_args()

    # Env checks (hard fail if missing)
    _ = require_env("OPENROUTER_API_KEY")
    _ = require_env("STABILITY_API_KEY")

    # IO
    outdir = Path(args.outdir)
    photos_dir = outdir / "photos"
    outdir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    # LLM client
    client = openrouter_client()
    model = DEFAULT_MODEL

    # CSV index
    index_path = outdir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "email", "phone", "role", "city", "country", "file", "photo"])

    rnd = random.Random(args.seed)

    for i in range(args.n):
        role = rnd.choice(ROLES)
        city = rnd.choice(CITIES)
        country = rnd.choice(COUNTRIES)

        # 1) Ask LLM for full CV JSON
        cv_json = gen_cv_json(
            client=client,
            model=model,
            seed=args.seed + i,
            role_hint=role,
            city_hint=city,
            country_hint=country
        )

        # 2) Generate photo
        slug = slugify(cv_json["name"])
        photo_path = photos_dir / f"{slug}_{i+1:02d}.jpg"
        stability_headshot(name=cv_json["name"], role=cv_json["role"], seed=args.seed + i, out_path=photo_path)

        # 3) Render PDF
        pdf_name = f"cv_{slug}_{i+1:02d}.pdf"
        pdf_path = outdir / pdf_name
        render_pdf(pdf_path, cv_json, photo_path)

        # 4) Index
        with index_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                cv_json["name"], cv_json["email"], cv_json["phone"],
                cv_json["role"], cv_json["city"], cv_json["country"],
                pdf_name, photo_path.name
            ])

        # Gentle pacing to be nice to APIs
        time.sleep(0.4)

    print(f"Generated {args.n} CVs → {outdir}")
    print(f"Index: {index_path}")

if __name__ == "__main__":
    main()
