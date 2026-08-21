"""
HR Contract Extraction Pipeline - PDF(s) -> Local LLM -> Validate -> Excel
----------------------------------------------------------------------------
Pipeline:
    1. PDF text extraction (+ PaddleOCR fallback for scanned/signed pages)
    2. Local LLM extraction (via Ollama) of a 20-field contract schema
    3. Field-level validation + review flagging
    4. Append the resulting row to a master Excel workbook (creates it on
       first run, appends a new row on every run after that)

Each run processes ONE OR TWO documents (e.g. an MSA and/or an NDA) for a
single partner and writes ONE combined row to the Excel sheet. Run it again
for the next partner/contract - rows accumulate in the same file.

PREREQUISITES:
    1. Install Ollama:      https://ollama.com/download
    2. Pull a model:        ollama pull phi4-mini     (recommended - llama3.2:1b
                             is too weak for reliable 20-field extraction)
    3. Install deps:        pip install -r requirements.txt
       (first run also auto-downloads PaddleOCR's model files, one-time only)

USAGE:
    # Single document
    python extract_to_excel.py --msa "path/to/msa.pdf"

    # MSA + NDA for the same partner, merged into one row
    python extract_to_excel.py --msa "msa.pdf" --nda "nda.pdf"

    # Choose model / output file
    python extract_to_excel.py --msa "msa.pdf" --model phi4-mini --output contracts.xlsx
"""

import sys
import os
import json
import re
import argparse
import time
import difflib
import subprocess
from pathlib import Path
from datetime import datetime

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

import pdfplumber
import fitz  # PyMuPDF - renders scanned pages as images for OCR
from paddleocr import PaddleOCR
import onnxruntime as ort
import ollama
from pydantic import BaseModel, ValidationError, field_validator

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# YOUR OWN COMPANY NAME + ADDRESS - set this once
# ---------------------------------------------------------------------------
OWN_COMPANY_NAME = "OneNextGen Automation Private Limited"
OWN_COMPANY_ALIASES = ["OneNextGen Automation", "OneNextGen", "OneNextGenAutomation"]
OWN_COMPANY_ADDRESS = (
    "No.359, 2nd Floor, 8thMain, 4th phase,7th block, Bsk 3rd stage, "
    "Gururaja Layout, Bengaluru - 560085"
)

# ---------------------------------------------------------------------------
# 14-FIELD SCHEMA (matches the exact column set requested)
# ---------------------------------------------------------------------------
SCHEMA_FIELDS = {
    "partner_name": "Legal name of the OTHER company (never our own company)",
    "partner_address": "Registered/official address of the partner company",
    "partner_contact_email": "Contact email address for the partner",
    "partner_contact_phone": "Contact phone number for the partner",
    "penalty_clause": "Y/N plus a one-sentence summary of any penalty clause, or 'N' if none present",
    "payment_terms": "Summary of payment terms (fees, milestones, currency, due days)",
    "payment_due_days": "ONLY the number of days payment is due within, digits only, e.g. '60'",
    "replacement_duration": "Time allowed to replace resource/goods/services if applicable",
    "taxation_and_billing": "Summary of tax responsibility and billing arrangement",
    "absconding_early_leave_clause": "Summary of any payment clause on absconding/early leave, or null",
    "term_start_date": "Contract start date",
    "term_end_date": "Contract end date (calculate from duration if needed)",
    "document_signed": "Y if there is clear evidence of an actual signature, else N",
    "invoice_to_be_sent_to": "Name/address/email invoices should be sent to",
    "governing_law": "State/country whose laws govern the agreement, e.g. 'Maharashtra, India'",
    "jurisdiction": "City/location for dispute resolution, e.g. 'Bangalore, India'",
    "confidentiality_clause": "Short phrase naming the core confidentiality obligation",
    "termination_clause": "Short phrase on how/when either party may terminate",
    "notice_period": "ONLY the number of days' notice required to terminate, digits only, e.g. '15'",
    "remarks": "Anything else notable that doesn't fit another field",
}

# Maps each schema key to the EXACT column header text requested for Excel.
FIELD_DISPLAY_NAMES = {
    "partner_name": "Partner Name",
    "partner_address": "Partner Address",
    "partner_contact_email": "Partner Contact Email",
    "partner_contact_phone": "Partner Contact Phone No.",
    "penalty_clause": "Penalty Clause? (Y/N & Details)",
    "payment_terms": "Payment Terms",
    "payment_due_days": "Payment Due Days",
    "replacement_duration": "Replacement Duration",
    "taxation_and_billing": "Taxation and Billing",
    "absconding_early_leave_clause": "Payment Clause on Absconding / Early Leave",
    "term_start_date": "Term of Agreement - Start Date",
    "term_end_date": "Term of Agreement - End Date",
    "document_signed": "Document Signed (Y/N)",
    "invoice_to_be_sent_to": "Invoice to be Sent To",
    "governing_law": "Governing Law",
    "jurisdiction": "Jurisdiction",
    "confidentiality_clause": "Confidentiality Clause",
    "termination_clause": "Termination Clause",
    "notice_period": "Notice Period",
    "remarks": "Remarks",
}


# ---------------------------------------------------------------------------
# PADDLEOCR ENGINE (loaded once, reused across all pages/documents)
# ---------------------------------------------------------------------------
def _prepare_onnx_models() -> dict[str, str]:
    """Convert PaddleOCR's cached Paddle models to ONNX files once."""
    model_root = Path.home() / ".paddleocr" / "whl"
    model_dirs = {
        "det_model_dir": next(model_root.glob("det/en/*"), None),
        "rec_model_dir": next(model_root.glob("rec/en/*"), None),
        "cls_model_dir": next(model_root.glob("cls/*"), None),
    }
    if any(model_dir is None for model_dir in model_dirs.values()):
        print("Downloading PaddleOCR models for ONNX conversion...")
        PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        model_dirs = {
            "det_model_dir": next(model_root.glob("det/en/*"), None),
            "rec_model_dir": next(model_root.glob("rec/en/*"), None),
            "cls_model_dir": next(model_root.glob("cls/*"), None),
        }

    onnx_models = {}
    for model_type, model_dir in model_dirs.items():
        if model_dir is None:
            raise FileNotFoundError(f"PaddleOCR {model_type} was not downloaded")
        onnx_path = model_dir.with_suffix(".onnx")
        if not onnx_path.exists():
            subprocess.run(
                [
                    str(Path(sys.executable).parent / "Scripts" / "paddle2onnx.exe"),
                    "--model_dir",
                    str(model_dir),
                    "--model_filename",
                    "inference.pdmodel",
                    "--params_filename",
                    "inference.pdiparams",
                    "--save_file",
                    str(onnx_path),
                    "--opset_version",
                    "11",
                ],
                check=True,
            )
        onnx_models[model_type] = str(onnx_path)
    return onnx_models


print("Loading PaddleOCR engine with ONNX Runtime (first run prepares model files, please wait)...")
_onnx_models = _prepare_onnx_models()
_paddle_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_onnx=True,
    **_onnx_models,
    show_log=False,
)


# ---------------------------------------------------------------------------
# PHONE NUMBER FALLBACK (regex over raw doc text)
# ---------------------------------------------------------------------------
# Small local LLMs frequently miss phone numbers buried in letterheads,
# footers, or signature blocks - especially after OCR. Before giving up and
# flagging "No contact phone found", scan the raw extracted text directly for
# common Indian phone formats: 10-digit mobiles (optionally spaced like
# "78160 63200"), and numbers with a +91 / 0 STD prefix or landline pattern.
PHONE_PATTERN = re.compile(
    r"(?:\+91[\-\s]?|0)?"          # optional +91 or leading 0
    r"(?:\d{5}[\-\s]\d{5}"         # e.g. 78160 63200 / 78160-63200
    r"|\d{10}"                     # e.g. 7816063200
    r"|\d{2,4}[\-\s]\d{6,8})"      # e.g. 080-12345678 (STD code + landline)
)


def find_phone_fallback(document_text: str) -> str | None:
    """Returns the first plausible phone number found via regex, or None."""
    for match in PHONE_PATTERN.finditer(document_text):
        digits = re.sub(r"\D", "", match.group(0))
        # keep only real-looking phone lengths (10 mobile, or 11-12 with STD/country code)
        if 10 <= len(digits) <= 12:
            return match.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# STAGE 1: PDF TEXT EXTRACTION (+ OCR fallback)
# ---------------------------------------------------------------------------
def ocr_page_with_paddle(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    """Renders one PDF page as a high-res image and runs PaddleOCR on it."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(dpi=dpi)
    img_path = f"_temp_ocr_page_{page_number}.png"
    pix.save(img_path)
    doc.close()

    result = _paddle_engine.ocr(img_path, cls=True)
    os.remove(img_path)

    if not result or not result[0]:
        return ""

    lines = result[0]
    lines_sorted = sorted(lines, key=lambda line: line[0][0][1])  # sort by y-coord
    text = "\n".join(line[1][0] for line in lines_sorted)
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text page by page; falls back to PaddleOCR for pages with
    no real text layer (scanned/signed documents)."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 20:
                print(f"  [OCR] Page {i + 1}/{num_pages} has no text layer - running PaddleOCR...")
                text = ocr_page_with_paddle(pdf_path, i)
                print(f"  [OCR] Page {i + 1}/{num_pages}: extracted {len(text.strip())} chars via OCR")
            full_text += text + "\n"
    return full_text


# ---------------------------------------------------------------------------
# STAGE 2: LLM EXTRACTION (via local Ollama)
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT_TEMPLATE = """You are a precise contract-data extraction engine. Extract ONLY information that is explicitly present in the document below. Never guess, infer, or invent values.

CONTEXT: This document is a contract between two companies:
- OUR COMPANY: "{own_company_name}" (may also appear as: {own_company_aliases})
- THE PARTNER: the other company named in the agreement - this is whoever is NOT our company, regardless of which party's letterhead the document uses or how it is labeled ("Vendor", "Service Provider", "Party A/B", etc.)

TASK: Read the entire document, identify both company names, confirm which one is the Partner, then extract these fields using EXACTLY these keys:

{field_descriptions}

OUTPUT RULES:
1. Return ONLY a single valid JSON object with EXACTLY these {num_fields} keys. No markdown, no commentary, no text before or after.
2. Every value must be a plain string, or null if the field is not explicitly stated.
3. Do NOT extract "{own_company_name}" or its aliases into any partner_* field, even if it appears first or on the letterhead.
4. For dates: return in DD-MMM-YYYY format if a specific date is given; if only a duration is stated (e.g. "12 months from the start date"), calculate the end date.
5. SHORT-ANSWER FIELDS - "payment_terms", "taxation_and_billing", "absconding_early_leave_clause", "confidentiality_clause", "termination_clause": each must be a SHORT PHRASE, ideally under 8 words, NEVER a full sentence copied from the document. Think "spreadsheet cell", not "summary paragraph". Examples: payment_terms -> "45 Days" (not "Payment to the Vendor within Forty-Five (45) days from the date of receipt of the invoice."); taxation_and_billing -> "Vendor responsible for PF, ESI, Gratuity" (not the full clause); absconding_early_leave_clause -> "No payment for No shows or client non-payments"; confidentiality_clause -> "Mutual NDA - no disclosure to third parties"; termination_clause -> "45 days' written notice to terminate". If you can't compress it into a short phrase, extract just the key term/number and put any extra nuance in "remarks" instead.
6. For "penalty_clause": start with "Y - " followed by a SHORT PHRASE (under 8 words) if a penalty clause exists (e.g. "Y - Violations of any applicable laws"), or just "N" if there is none. Do not paste the full clause sentence.
7. For document_signed: "Y" only if there is clear evidence of an actual signature (not an empty signature block/placeholder). Otherwise "N".
8. For "payment_due_days" and "notice_period": digits only, no units or extra text (e.g. "60", not "60 days" or "within 60 days").
9. For "governing_law": extract the state/country whose laws govern the agreement (e.g. "Maharashtra, India"), usually found in a "Governing Law" clause.
10. For "jurisdiction": extract the city/location where disputes/arbitration are to be settled (e.g. "Bangalore, India") - this may be different from governing_law.
11. If a field is genuinely absent, set it to null - never fabricate a plausible-sounding value.

DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"

Return the JSON object now.
"""


def build_field_descriptions() -> str:
    return "\n".join(f'- "{k}": {v}' for k, v in SCHEMA_FIELDS.items())


def extract_fields_with_llm(document_text: str, model: str) -> tuple[str, float]:
    """Send document text to a local Ollama model and get back raw JSON text."""
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        own_company_name=OWN_COMPANY_NAME,
        own_company_aliases=", ".join(OWN_COMPANY_ALIASES),
        field_descriptions=build_field_descriptions(),
        num_fields=len(SCHEMA_FIELDS),
        document_text=document_text
    )

    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 32768,     # generous window so long documents aren't truncated
            "num_predict": 3072
        }
    )
    elapsed = time.time() - start
    return response["message"]["content"], elapsed


def clean_llm_json_output(raw_output: str) -> dict:
    """Strips markdown fences / stray text and parses safely."""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse LLM output as JSON: {e}\n\nRaw output:\n{raw_output}")


def coerce_all_fields_to_strings(parsed_dict: dict) -> dict:
    """Flattens any unexpected type into a plain string so validation doesn't
    crash. Small local models sometimes return numbers (e.g. a numeric field
    returned as 30
    instead of "30"), booleans, or nested objects/lists instead of the plain
    strings the schema expects."""
    cleaned = {}
    for key, value in parsed_dict.items():
        if isinstance(value, bool):
            cleaned[key] = "Y" if value else "N"
        elif isinstance(value, (dict, list)):
            cleaned[key] = json.dumps(value)
        elif isinstance(value, (int, float)):
            cleaned[key] = str(value)
        else:
            cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# STAGE 3: SCHEMA VALIDATION (Pydantic)
# ---------------------------------------------------------------------------
class PartnerAgreementRecord(BaseModel):
    partner_name: str | None = None
    partner_address: str | None = None
    partner_contact_email: str | None = None
    partner_contact_phone: str | None = None
    penalty_clause: str | None = None
    payment_terms: str | None = None
    payment_due_days: str | None = None
    replacement_duration: str | None = None
    taxation_and_billing: str | None = None
    absconding_early_leave_clause: str | None = None
    term_start_date: str | None = None
    term_end_date: str | None = None
    document_signed: str | None = None
    invoice_to_be_sent_to: str | None = None
    governing_law: str | None = None
    jurisdiction: str | None = None
    confidentiality_clause: str | None = None
    termination_clause: str | None = None
    notice_period: str | None = None
    remarks: str | None = None

    @field_validator("document_signed")
    @classmethod
    def normalize_yn(cls, v):
        if v is None:
            return v
        v_upper = v.strip().upper()
        if v_upper.startswith("Y"):
            return "Y"
        if v_upper.startswith("N"):
            return "N"
        return v


# ---------------------------------------------------------------------------
# STAGE 4: FIELD-LEVEL VALIDATION + REVIEW FLAGGING
# ---------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"^[\w\.\-+]+@[\w\-]+\.[a-zA-Z]{2,}$")


def validate_record(record: PartnerAgreementRecord) -> tuple[dict, list[str]]:
    data = record.model_dump()
    review_reasons = []

    if not data.get("partner_name"):
        review_reasons.append("Missing Partner Name")
    else:
        partner_name_lower = data["partner_name"].lower()
        own_names_to_check = [OWN_COMPANY_NAME.lower()] + [a.lower() for a in OWN_COMPANY_ALIASES]
        if any(own_name in partner_name_lower for own_name in own_names_to_check):
            review_reasons.append(
                f"WARNING: partner_name looks like OUR OWN company ('{data['partner_name']}') - verify manually"
            )

    if not data.get("partner_address"):
        review_reasons.append("Missing Partner Address")

    email = data.get("partner_contact_email")
    if email:
        first_email = email.split("/")[0].strip()
        if not EMAIL_PATTERN.match(first_email):
            review_reasons.append(f"Email format looks invalid: {email}")
    else:
        review_reasons.append("No contact email found in document")

    if not data.get("partner_contact_phone"):
        review_reasons.append("No contact phone found in document")

    start_date_raw = data.get("term_start_date")
    if start_date_raw:
        try:
            parsed_start = date_parser.parse(start_date_raw, dayfirst=True)
            data["term_start_date"] = parsed_start.strftime("%d-%b-%y")
            end_date_raw = data.get("term_end_date")
            if not end_date_raw:
                calculated_end = parsed_start + relativedelta(years=1)
                data["term_end_date"] = calculated_end.strftime("%d-%b-%y")
                review_reasons.append("End Date auto-calculated (assumed 1-year term) - verify")
            else:
                try:
                    parsed_end = date_parser.parse(end_date_raw, dayfirst=True)
                    data["term_end_date"] = parsed_end.strftime("%d-%b-%y")
                    if parsed_end <= parsed_start:
                        review_reasons.append(
                            f"Term End Date ({data['term_end_date']}) is not after Start Date - verify"
                        )
                except (ValueError, OverflowError):
                    review_reasons.append(f"Could not parse End Date (may be an invalid date like 31-Feb): {end_date_raw}")
                    data["term_end_date"] = None
        except (ValueError, OverflowError):
            review_reasons.append(f"Could not parse Start Date: {start_date_raw}")
    else:
        review_reasons.append("Missing Term Start Date")

    if data.get("document_signed") not in ("Y", "N"):
        review_reasons.append(f"Document Signed value unclear: {data.get('document_signed')}")

    if data.get("penalty_clause") is None:
        review_reasons.append("Penalty Clause not found - confirm manually")

    if not data.get("replacement_duration"):
        review_reasons.append("Missing Replacement Duration")
    if not data.get("taxation_and_billing"):
        review_reasons.append("Missing Taxation and Billing")
    if not data.get("invoice_to_be_sent_to"):
        review_reasons.append("Missing Invoice-To details")

    # New field checks
    payment_due_days = data.get("payment_due_days")
    if payment_due_days:
        digits_only = re.sub(r"[^\d]", "", str(payment_due_days))
        if not digits_only:
            review_reasons.append(f"Payment Due Days not numeric: {payment_due_days}")
        else:
            data["payment_due_days"] = digits_only
    else:
        review_reasons.append("Missing Payment Due Days")

    if not data.get("governing_law"):
        review_reasons.append("Missing Governing Law")
    if not data.get("jurisdiction"):
        review_reasons.append("Missing Jurisdiction")
    if not data.get("confidentiality_clause"):
        review_reasons.append("Missing Confidentiality Clause")
    if not data.get("termination_clause"):
        review_reasons.append("Missing Termination Clause")

    notice_period = data.get("notice_period")
    if notice_period:
        digits_only = re.sub(r"[^\d]", "", str(notice_period))
        if not digits_only:
            review_reasons.append(f"Notice Period not numeric: {notice_period}")
        else:
            data["notice_period"] = digits_only
    else:
        review_reasons.append("Missing Notice Period")

    data["needs_review"] = len(review_reasons) > 0
    data["review_reasons"] = "; ".join(review_reasons)
    return data, review_reasons


# ---------------------------------------------------------------------------
# STAGE 5: MERGE MULTIPLE DOCUMENTS (e.g. MSA + NDA) INTO ONE ROW
# ---------------------------------------------------------------------------
# When two documents genuinely disagree on a field (which is expected for an
# MSA vs an NDA - they cover different clauses), we don't want to jam both
# full texts into one Excel cell. Instead we pick ONE clean value per field
# based on this priority order, and record every disagreement in the review
# notes so nothing is silently lost - just kept out of the cell itself.
FIELD_SOURCE_PRIORITY = {
    # Commercial/operational terms: MSA is authoritative
    "payment_terms": ["MSA", "NDA"],
    "payment_due_days": ["MSA", "NDA"],
    "replacement_duration": ["MSA", "NDA"],
    "taxation_and_billing": ["MSA", "NDA"],
    "absconding_early_leave_clause": ["MSA", "NDA"],
    "penalty_clause": ["MSA", "NDA"],
    "invoice_to_be_sent_to": ["MSA", "NDA"],
    "governing_law": ["MSA", "NDA"],
    "jurisdiction": ["MSA", "NDA"],
    # Confidentiality/termination terms: NDA is authoritative
    "confidentiality_clause": ["NDA", "MSA"],
    "termination_clause": ["NDA", "MSA"],
    "notice_period": ["NDA", "MSA"],
    # Everything else (dates, partner identity, etc.): no strong prior -
    # first source wins for the cell, conflict still logged.
}


def _is_own_company(value: str | None) -> bool:
    if not value:
        return False
    value_lower = value.lower()
    own_names = [OWN_COMPANY_NAME.lower()] + [a.lower() for a in OWN_COMPANY_ALIASES]
    return any(own_name in value_lower for own_name in own_names)


def _normalize_address(text: str) -> str:
    """Lowercases and strips punctuation/whitespace so OCR noise ('Bsk' vs
    'BSk', stray commas, extra spaces) doesn't break comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_own_address(value: str | None, threshold: float = 0.6) -> bool:
    """Fuzzy-matches a candidate address against OWN_COMPANY_ADDRESS. Uses a
    similarity ratio (not exact match) because OCR'd text often has small
    typos ('Bsk' vs 'BSk', missing spaces, stray characters). Also checks
    whether the shorter address is essentially a substring of the longer
    one, since the LLM sometimes appends extra boilerplate ("...hereinafter
    referred to as the Second Party...") after a correctly-matched address."""
    if not value or not OWN_COMPANY_ADDRESS:
        return False
    candidate = _normalize_address(value)
    own = _normalize_address(OWN_COMPANY_ADDRESS)
    if not candidate or not own:
        return False

    # Whole-string similarity
    ratio = difflib.SequenceMatcher(None, candidate, own).ratio()
    if ratio >= threshold:
        return True

    # Substring check: does the candidate contain a near-match of our
    # address as a prefix/segment (handles trailing boilerplate text)?
    shorter, longer = (own, candidate) if len(own) <= len(candidate) else (candidate, own)
    if len(shorter) >= 15:
        window = longer[:len(shorter) + 20]
        return difflib.SequenceMatcher(None, shorter, window).ratio() >= threshold

    return False


COMPANY_NAME_PATTERN = re.compile(
    r"[A-Z][A-Za-z0-9&,\.\(\)\s]{2,70}"
    r"(?:Private Limited|Pvt\.?\s*Ltd\.?|Limited|LLP|LLC|Inc\.?|Corporation|Corp\.?)",
)

# Common contract-preamble words that end up glued onto a company name when
# the regex grabs everything capitalized before a "Private Limited"/"LLP"
# suffix. These get stripped off the front of any candidate before it's
# counted, since they're clause structure, not part of the name.
NAME_BOILERPLATE_PREFIXES = {
    "between", "and", "whereas", "witnesseth", "this", "agreement", "made",
    "entered", "into", "on", "day", "of", "at", "dated", "party", "parties",
    "first", "second", "hereinafter", "by", "among", "amongst", "the",
    "company", "consultant", "consultants", "client", "vendor",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def _strip_boilerplate_prefix(candidate: str) -> str:
    """Repeatedly drops leading words that are contract boilerplate (or an
    OCR-typo'd variant of it, e.g. 'BETWEENR' for 'BETWEEN') until the
    candidate starts with what's actually likely to be a company name."""
    words = candidate.split()
    while words:
        first = re.sub(r"[^a-z]", "", words[0].lower())
        is_boilerplate = first in NAME_BOILERPLATE_PREFIXES or any(
            len(first) >= 4 and difflib.SequenceMatcher(None, first, stopword).ratio() >= 0.82
            for stopword in NAME_BOILERPLATE_PREFIXES
        )
        # A bare 4-digit year (e.g. "2024") is also boilerplate here.
        is_year = re.fullmatch(r"\d{4}", words[0]) is not None
        if is_boilerplate or is_year:
            words.pop(0)
        else:
            break
    return " ".join(words)


def guess_partner_name_from_text(document_text: str) -> str | None:
    """Fallback for when the LLM extracts our own company as the partner:
    scans the raw document text for company-name-shaped strings (ending in
    'Private Limited', 'LLP', 'Inc', etc.), strips leading contract-preamble
    words that the regex over-captured, excludes our own company/aliases,
    and returns the most frequently mentioned remaining candidate. This is a
    heuristic, not a guarantee - the record still gets flagged for manual
    verification."""
    candidates = [m.strip() for m in COMPANY_NAME_PATTERN.findall(document_text)]
    own_names = [OWN_COMPANY_NAME.lower()] + [a.lower() for a in OWN_COMPANY_ALIASES]

    counts: dict[str, int] = {}
    for c in candidates:
        c_clean = re.sub(r"\s+", " ", c).strip(" ,.")
        c_clean = _strip_boilerplate_prefix(c_clean)
        if len(c_clean) < 5:
            continue
        if any(own in c_clean.lower() for own in own_names):
            continue
        counts[c_clean] = counts.get(c_clean, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)


def merge_records(dicts: list[dict], labels: list[str]) -> tuple[dict, list[str]]:
    """Field-by-field merge across N source documents. Whichever document has
    a non-null value wins; if two documents DISAGREE on a field, ONE clean
    value is kept in the cell (per FIELD_SOURCE_PRIORITY, else first source),
    and the disagreement is recorded as a conflict for the review column -
    never concatenated into the cell itself."""
    dicts = [dict(d) for d in dicts]  # don't mutate caller's dicts

    # Safety net: if a source's partner_name is actually OUR OWN company (a
    # known LLM failure mode), drop that source's partner_* fields entirely
    # so a correct value from the other source can win. partner_address is
    # also checked independently, since the name and address don't always
    # get confused together.
    for label, d in zip(labels, dicts):
        if _is_own_company(d.get("partner_name")):
            for field in ("partner_name", "partner_address",
                          "partner_contact_email", "partner_contact_phone"):
                d[field] = None
        elif _is_own_address(d.get("partner_address")):
            d["partner_address"] = None

    merged = {}
    conflicts = []

    for key in SCHEMA_FIELDS:
        present_values = [
            (label, d.get(key)) for label, d in zip(labels, dicts)
            if d.get(key) is not None and str(d.get(key)).strip() != ""
        ]

        if not present_values:
            merged[key] = None
            continue

        distinct = {str(v).strip().lower() for _, v in present_values}
        if len(distinct) == 1:
            merged[key] = present_values[0][1]
            continue

        # Genuine disagreement: pick one clean value for the cell.
        priority = FIELD_SOURCE_PRIORITY.get(key, labels)
        by_label = dict(present_values)
        chosen_label = next((lbl for lbl in priority if lbl in by_label), present_values[0][0])
        merged[key] = by_label[chosen_label]

        conflicts.append(
            f"'{key}' differs across sources (used {chosen_label}): " +
            ", ".join(f"{label}='{val}'" for label, val in present_values)
        )

    return merged, conflicts


def process_single_document(pdf_path: str, model: str, label: str) -> dict:
    """Runs Stages 1-3 on one document and returns the parsed field dict."""
    print(f"\n{'=' * 60}\n{label}: {pdf_path}\n{'=' * 60}")

    doc_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(doc_text)} characters.")

    print(f"Sending to local LLM (model: {model})...")
    try:
        raw_output, elapsed = extract_fields_with_llm(doc_text, model)
    except Exception as e:
        print(f"[ERROR] Could not reach Ollama or run the model: {e}")
        sys.exit(1)
    print(f"Responded in {elapsed:.1f} seconds.")

    try:
        parsed_dict = clean_llm_json_output(raw_output)
        parsed_dict = coerce_all_fields_to_strings(parsed_dict)
        record = PartnerAgreementRecord(**parsed_dict)
    except (ValueError, ValidationError) as e:
        print(f"[FAILED] {label} extraction/validation failed: {e}")
        sys.exit(1)

    result = record.model_dump()

    # Known LLM failure mode: extracted our own company as the partner.
    # Try to recover the real partner name by scanning the raw document
    # text for other company-name-shaped strings before giving up.
    if _is_own_company(result.get("partner_name")):
        guess = guess_partner_name_from_text(doc_text)
        if guess:
            print(f"  [CORRECTED] partner_name was our own company - "
                  f"replaced with best guess from document text: '{guess}'")
            result["partner_name"] = guess
        else:
            print("  [WARNING] partner_name was our own company and no "
                  "alternate candidate was found in the text - clearing it.")
            result["partner_name"] = None
        # Address/email/phone extracted alongside a wrong partner_name are
        # also likely to be OUR OWN details rather than the partner's -
        # clear them too rather than risk publishing wrong contact info.
        for field in ("partner_address", "partner_contact_email", "partner_contact_phone"):
            if _is_own_company(result.get(field)):
                result[field] = None

    # Independent check: partner_address can be OUR OWN address even when
    # partner_name was extracted correctly (the two don't always get
    # confused together). Fuzzy-matched to survive OCR typos.
    if _is_own_address(result.get("partner_address")):
        print(f"  [CORRECTED] partner_address matched our own company's "
              f"address - clearing it (was: '{result['partner_address']}')")
        result["partner_address"] = None

    # Fallback: if the LLM didn't find a phone number, try a regex scan of
    # the raw document text before accepting "not found".
    if not result.get("partner_contact_phone"):
        fallback_phone = find_phone_fallback(doc_text)
        if fallback_phone:
            print(f"  [FALLBACK] LLM missed phone number - regex found: {fallback_phone}")
            result["partner_contact_phone"] = fallback_phone

    return result


# ---------------------------------------------------------------------------
# STAGE 6: WRITE / APPEND TO MASTER EXCEL WORKBOOK
# ---------------------------------------------------------------------------
# Column order: S.No. first, then exactly the requested 14 columns (with
# your exact header text), then two small tracking columns at the end
# (Needs Review / Review Notes) so flagged rows can still be filtered on
# the dashboard - remove them from EXCEL_COLUMNS below if you don't want
# them in the sheet at all.
EXCEL_COLUMNS = ["s_no"] + list(SCHEMA_FIELDS.keys()) + ["needs_review", "review_reasons"]

COLUMN_HEADERS = {
    "s_no": "S.No.",
    **FIELD_DISPLAY_NAMES,
    "needs_review": "Needs Review",
    "review_reasons": "Review Notes",
}

HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(name="Arial", size=11, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial", size=10)
REVIEW_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
THIN_BORDER = Border(*(Side(style="thin", color="D9D9D9"),) * 4)


def ensure_workbook(path: str):
    """Creates the master workbook with a styled header row if it doesn't
    exist yet; otherwise loads the existing one so we can append."""
    if os.path.exists(path):
        wb = load_workbook(path)
        ws = wb["Contracts"]
        return wb, ws

    wb = Workbook()
    ws = wb.active
    ws.title = "Contracts"

    for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=COLUMN_HEADERS.get(col_name, col_name))
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 28

    widths = {
        "s_no": 6,
        "partner_name": 26, "partner_address": 32, "partner_contact_email": 24,
        "partner_contact_phone": 16, "penalty_clause": 34, "payment_terms": 28,
        "payment_due_days": 14, "replacement_duration": 18, "taxation_and_billing": 28,
        "absconding_early_leave_clause": 32, "term_start_date": 16, "term_end_date": 16,
        "document_signed": 12, "invoice_to_be_sent_to": 24,
        "governing_law": 18, "jurisdiction": 18, "confidentiality_clause": 30,
        "termination_clause": 30, "notice_period": 14,
        "remarks": 30, "needs_review": 12, "review_reasons": 40,
    }
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = widths.get(col_name, 18)

    return wb, ws


def append_row_to_excel(path: str, row_dict: dict):
    wb, ws = ensure_workbook(path)
    next_row = ws.max_row + 1
    row_dict = dict(row_dict)
    row_dict["s_no"] = next_row - 1  # header is row 1, so row 2 = S.No. 1

    for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
        value = row_dict.get(col_name)
        if isinstance(value, bool):
            value = "Y" if value else "N"
        cell = ws.cell(row=next_row, column=col_idx, value=value)
        cell.font = BODY_FONT
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.border = THIN_BORDER
        if row_dict.get("needs_review"):
            cell.fill = REVIEW_FILL

    wb.save(path)
    print(f"\nAppended row {next_row - 1} to '{path}' (sheet: Contracts)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract contract fields from one or more PDFs and append the merged record to an Excel workbook."
    )
    parser.add_argument("--msa", help="Path to the MSA PDF file")
    parser.add_argument("--nda", help="Path to the NDA PDF file")
    parser.add_argument(
        "--model", default="phi4-mini",
        help="Ollama model to use (default: phi4-mini). Other options: qwen3:8b, llama3.2:1b (not recommended)"
    )
    parser.add_argument(
        "--output", default="contracts_master.xlsx",
        help="Path to the master Excel workbook (created if it doesn't exist, appended to otherwise)"
    )
    args = parser.parse_args()

    if not args.msa and not args.nda:
        parser.error("Provide at least one of --msa or --nda")

    labels, dicts, source_files = [], [], []
    if args.msa:
        dicts.append(process_single_document(args.msa, args.model, "MSA"))
        labels.append("MSA")
        source_files.append(os.path.basename(args.msa))
    if args.nda:
        dicts.append(process_single_document(args.nda, args.model, "NDA"))
        labels.append("NDA")
        source_files.append(os.path.basename(args.nda))

    if len(dicts) > 1:
        print(f"\n{'=' * 60}\nMerging {' + '.join(labels)} into one combined record...\n{'=' * 60}")
        merged_dict, conflicts = merge_records(dicts, labels)
        if conflicts:
            print(f"\n[CONFLICTS FOUND] {len(conflicts)} field(s) differ between sources:")
            for c in conflicts:
                print(f"  - {c}")
        else:
            print("No conflicts - sources agree on all overlapping fields.")
    else:
        merged_dict, conflicts = dicts[0], []

    try:
        merged_record = PartnerAgreementRecord(**merged_dict)
    except ValidationError as e:
        print(f"Merged record failed schema validation:\n{e}")
        sys.exit(1)

    print(f"\n{'=' * 60}\nField Validation + Review Flagging...\n{'=' * 60}")
    final_record, review_reasons = validate_record(merged_record)

    if conflicts:
        review_reasons.extend(conflicts)
        final_record["needs_review"] = True
        final_record["review_reasons"] = "; ".join(review_reasons)

    final_record["processed_at"] = datetime.now().strftime("%d-%b-%y %H:%M")
    final_record["model_used"] = args.model
    final_record["source_files"] = " + ".join(source_files)

    print("\nFINAL RECORD:\n")
    print(json.dumps(final_record, indent=2))

    if review_reasons:
        print(f"\n[FLAGGED FOR REVIEW] {len(review_reasons)} issue(s) found:")
        for reason in review_reasons:
            print(f"  - {reason}")
    else:
        print("\nNo review flags - all fields look good.")

    append_row_to_excel(args.output, final_record)
    print(f"\nDone. Run 'streamlit run dashboard.py' to view all contracts on the dashboard.")


if __name__ == "__main__":
    main()