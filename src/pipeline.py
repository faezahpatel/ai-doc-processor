from pathlib import Path
import uuid, time, re, datetime as dt
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    spacy = None
    _NLP = None

class Config:
    MIN_CONFIDENCE_ROUTE = 0.88

def now_iso():
    return dt.datetime.utcnow().isoformat() + "Z"

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".pdf"]: return "application/pdf"
    if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]: return "image"
    if ext in [".txt"]: return "text"
    return "unknown"

def extract_text_pdf(path: Path) -> str:
    if not pdfplumber:
        return ""
    text_chunks = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            text_chunks.append(txt)
    return "\n".join(text_chunks)

def extract_tables_pdf(path: Path) -> List[pd.DataFrame]:
    if not pdfplumber:
        return []
    tables = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                except Exception:
                    df = pd.DataFrame(table)
                tables.append(df)
    return tables

def extract_text_image(path: Path) -> str:
    if not (Image and pytesseract):
        return ""
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_text(path: Path):
    mime = guess_mime(path)
    text, tables = "", []
    if mime == "application/pdf":
        text = extract_text_pdf(path)
        tables = extract_tables_pdf(path)
    elif mime == "image":
        text = extract_text_image(path)
    elif mime == "text":
        text = path.read_text(encoding="utf-8", errors="ignore")
    return text.strip(), tables

INVOICE_HINTS = ["invoice", "amount due", "bill to", "subtotal", "total", "tax"]
FORM_HINTS = ["form", "name:", "email", "phone", "address"]
CONTRACT_HINTS = ["agreement", "party", "contract", "effective date", "term"]

def classify_document(text: str) -> str:
    t = text.lower()
    score = {"invoice":0, "form":0, "contract":0, "unknown":0}
    score["invoice"] += sum(h in t for h in INVOICE_HINTS)
    score["form"] += sum(h in t for h in FORM_HINTS)
    score["contract"] += sum(h in t for h in CONTRACT_HINTS)
    cls = max(score, key=score.get)
    return cls if score[cls] > 0 else "unknown"

MONEY_RE = re.compile(r"(?:USD\s?)?\$\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/.-]){2}\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s*\d{4}")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")

def extract_entities_regex(text: str):
    ents = {
        "MONEY": MONEY_RE.findall(text),
        "DATE": DATE_RE.findall(text),
        "EMAIL": EMAIL_RE.findall(text),
        "PHONE": PHONE_RE.findall(text),
    }
    return ents

def extract_entities_spacy(text: str):
    if not _NLP:
        return extract_entities_regex(text)
    doc = _NLP(text)
    ents = {}
    for e in doc.ents:
        ents.setdefault(e.label_, []).append(e.text)
    for k,v in extract_entities_regex(text).items():
        ents.setdefault(k, [])
        ents[k].extend(v)
    return ents

INVOICE_SCHEMA = {
    "company_name": {"required": True},
    "invoice_number": {"required": True},
    "invoice_date": {"required": True},
    "total_amount": {"required": True},
}

def map_invoice_fields(text: str, ents):
    fields = {}
    m = re.search(r"(?i)(?:company|vendor|bill from)[:\s]+([A-Za-z0-9&.,'\-\s]{3,})", text)
    if m: fields["company_name"] = m.group(1).strip()
    m = re.search(r"(?i)(invoice\s*(?:no|number|#)[:\s]*)([A-Za-z0-9-]+)", text)
    if m: fields["invoice_number"] = m.group(2).strip()
    fields["invoice_date"] = ents.get("DATE", [None])[0]
    money_list = ents.get("MONEY", [])
    fields["total_amount"] = max(money_list, default=None) if money_list else None
    return fields

def validate_fields(fields, schema):
    conf = {}
    valid = True
    for key, rule in schema.items():
        val = fields.get(key)
        req = rule.get("required", False)
        present = val not in (None, "", [], {})
        if req and not present:
            valid = False
            conf[key] = 0.0
        else:
            if key.endswith("date") and val:
                conf[key] = 0.9
            elif key.endswith("amount") and val:
                conf[key] = 0.92
            elif val:
                conf[key] = 0.85
            else:
                conf[key] = 0.0
    return valid, conf

def aggregate_confidence(conf_map):
    if not conf_map: return 0.0
    return float(np.mean(list(conf_map.values())))

def enrich(fields):
    VENDOR_DB = {"myOnsite Healthcare LLC":{"vendor_id":"VEND-001","domain":"healthcare"}}
    company = fields.get("company_name")
    if company in VENDOR_DB:
        fields.update(VENDOR_DB[company])
    return fields

def route_decision(doc_conf: float, business_critical: bool = False) -> str:
    if business_critical and doc_conf < 0.95:
        return "human_review"
    if doc_conf < Config.MIN_CONFIDENCE_ROUTE:
        return "human_review"
    return "auto_approve"

def process_document(path: Path):
    text, tables = extract_text(path)
    cls = classify_document(text) if text else "unknown"
    ents = extract_entities_spacy(text) if text else {}
    if cls == "invoice":
        mapped = map_invoice_fields(text, ents)
        valid, confmap = validate_fields(mapped, INVOICE_SCHEMA)
    else:
        mapped, valid, confmap = {"raw_excerpt": text[:500] if text else ""}, True, {"raw_excerpt":0.6 if text else 0.0}
    mapped = enrich(mapped)
    doc_conf = aggregate_confidence(confmap)
    route = route_decision(doc_conf, business_critical=False)
    return {
        "path": str(path),
        "class": cls,
        "fields": mapped,
        "field_confidence": confmap,
        "document_confidence": round(doc_conf,3),
        "route": route,
        "processed_at": now_iso(),
    }
