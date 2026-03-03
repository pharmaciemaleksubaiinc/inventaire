# inventaire.py
# pip install streamlit pandas openpyxl pdfplumber
# Optional PDF export:
#   pip install reportlab

import io
import re
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

# PDF read
try:
    import pdfplumber
    PDFPLUMBER_OK = True
except Exception:
    PDFPLUMBER_OK = False

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


DEFAULT_EXCLUDED_WORDS = {"APO"}  # add more in sidebar


# -----------------------------
# Parsing (your logic, safer)
# -----------------------------
def normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_med_cell(cell: object) -> Optional[Tuple[str, str]]:
    if cell is None:
        return None
    entry = str(cell).strip()
    if not entry:
        return None
    if not entry[0].isdecimal():
        return None

    lines = entry.splitlines()
    if len(lines) < 2:
        return None

    med_name = lines[1].strip()

    manufacturer_prefix = ""
    if len(lines) >= 4 and lines[3].strip():
        manufacturer_prefix = lines[3].split()[0].strip()

    return med_name, manufacturer_prefix


def parse_amount(cell: object) -> Optional[float]:
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None

    lines = text.splitlines()

    # primary: 3rd line first token
    if len(lines) >= 3:
        parts = lines[2].split()
        if parts:
            token = parts[0].replace(",", ".")
            try:
                return float(token)
            except Exception:
                pass

    # fallback: any number
    m = re.search(r"(-?\d+(?:[.,]\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None
    return None


def strip_prefix(med_name: str, manufacturer_prefix: str, excluded_words: set) -> str:
    words = med_name.split()
    first = words[0].upper() if words else ""
    if first and (first == manufacturer_prefix.upper() or first in excluded_words):
        med_name = " ".join(words[1:]).strip()
    return normalise_whitespace(med_name)


def aggregate_inventory(df: pd.DataFrame, med_col_idx: int, amt_col_idx: int, excluded_words: set):
    hashmap = {}
    parsed = 0
    skipped = 0

    med_col = df.iloc[1:, med_col_idx]
    amt_col = df.iloc[1:, amt_col_idx]

    for med_cell, amt_cell in zip(med_col, amt_col):
        pm = parse_med_cell(med_cell)
        if not pm:
            skipped += 1
            continue
        med_name, manufacturer_prefix = pm

        amount = parse_amount(amt_cell)
        if amount is None:
            skipped += 1
            continue

        med_name = strip_prefix(med_name, manufacturer_prefix, excluded_words)
        if not med_name:
            skipped += 1
            continue

        hashmap[med_name] = hashmap.get(med_name, 0.0) + float(amount)
        parsed += 1

    out = (
        pd.DataFrame(list(hashmap.items()), columns=["Name", "Amount"])
        .sort_values("Name")
        .reset_index(drop=True)
    )
    return out, parsed, skipped


# -----------------------------
# Auto-detect columns
# -----------------------------
def score_med_column(series: pd.Series, sample_n: int = 200) -> int:
    """
    Higher score = more rows look like your medication multi-line format.
    """
    score = 0
    for v in series.dropna().astype(str).head(sample_n):
        v = v.strip()
        if not v:
            continue
        # must start with digit
        if not v[0].isdecimal():
            continue
        lines = v.splitlines()
        # should have at least 2 lines, and line 2 (index1) should be non-empty
        if len(lines) >= 2 and lines[1].strip():
            score += 1
    return score


def score_amount_column(series: pd.Series, sample_n: int = 200) -> int:
    """
    Higher score = more values parse as an amount using your rules.
    """
    score = 0
    for v in series.dropna().head(sample_n):
        if parse_amount(v) is not None:
            score += 1
    return score


def auto_pick_columns(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Picks the best med column and amount column using scoring.
    """
    med_scores = []
    amt_scores = []

    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        med_scores.append((score_med_column(s), i))
        amt_scores.append((score_amount_column(s), i))

    med_scores.sort(reverse=True)  # highest first
    amt_scores.sort(reverse=True)

    best_med = med_scores[0][1]
    best_amt = amt_scores[0][1]

    # If it accidentally picks same column, choose next best amount
    if best_amt == best_med and len(amt_scores) > 1:
        best_amt = amt_scores[1][1]

    return best_med, best_amt


# -----------------------------
# Reading files
# -----------------------------
def read_pdf_to_dataframe(uploaded_file) -> pd.DataFrame:
    if not PDFPLUMBER_OK:
        raise RuntimeError("PDF support not installed. Add pdfplumber to requirements.txt.")

    all_rows: List[List[object]] = []
    max_cols = 0

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row is None:
                        continue
                    row = list(row)
                    max_cols = max(max_cols, len(row))
                    all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No tables detected in PDF. If it's a scanned PDF, you need OCR.")

    padded = []
    for row in all_rows:
        row = row + [None] * (max_cols - len(row))
        padded.append(row)

    return pd.DataFrame(padded)


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), "Loaded CSV"

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file), "Loaded Excel"

    if name.endswith(".pdf"):
        return read_pdf_to_dataframe(uploaded_file), "Loaded PDF (via pdfplumber)"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, or PDF.")


# -----------------------------
# Export PDF (optional)
# -----------------------------
def dataframe_to_simple_pdf(df: pd.DataFrame, title: str = "Inventaire réorganisé") -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("PDF export requires reportlab. Add 'reportlab' to requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    width, height = letter
    x = 0.75 * inch
    y = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    line_height = 12

    # Header
    header = f"{'Name':60}  {'Amount':>10}"
    c.drawString(x, y, header)
    y -= line_height
    c.drawString(x, y, "-" * 80)
    y -= line_height

    for _, row in df.iterrows():
        name = str(row["Name"])[:60]
        amt = f"{row['Amount']:.2f}"
        line = f"{name:60}  {amt:>10}"

        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 10)

        c.drawString(x, y, line)
        y -= line_height

    c.save()
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Inventaire — Nettoyage & Fusion", layout="wide")
st.title("Inventaire — Nettoyage & fusion (prefixes ignorés)")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Règles")
    excluded_text = st.text_input(
        "Prefixes à ignorer (séparés par virgules)",
        value=",".join(sorted(DEFAULT_EXCLUDED_WORDS)),
        help="Ex: APO,SANDOZ,TEVA"
    )
    excluded_words = {w.strip().upper() for w in excluded_text.split(",") if w.strip()}

    st.divider()
    st.header("Exports")
    export_pdf = st.checkbox("Générer aussi un PDF", value=False)
    if export_pdf and not REPORTLAB_OK:
        st.warning("PDF export needs reportlab. Add 'reportlab' to requirements.txt.")

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw, note = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(note)

# Auto pick columns
med_col_idx, amt_col_idx = auto_pick_columns(df_raw)
st.caption(f"Auto-detected columns → Medication: {med_col_idx} | Amount: {amt_col_idx}")

# Run aggregation
out, parsed, skipped = aggregate_inventory(df_raw, med_col_idx, amt_col_idx, excluded_words)

st.subheader("Inventaire réorganisé (prefixes supprimés + totaux)")
st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")
st.dataframe(out, use_container_width=True)

# Excel download
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Inventaire")

st.download_button(
    "Télécharger Excel (.xlsx)",
    data=excel_buf.getvalue(),
    file_name="inventaire_reorganise.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# PDF download (optional)
if export_pdf:
    try:
        pdf_bytes = dataframe_to_simple_pdf(out, title="Inventaire réorganisé (prefixes ignorés)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_reorganise.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
