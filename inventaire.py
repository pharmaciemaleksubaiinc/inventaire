# inventaire.py
# pip install streamlit pandas openpyxl pdfplumber pdfminer.six
# Optional PDF export: pip install reportlab

import io
import re
from typing import List, Optional, Tuple

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


# -----------------------------
# Settings
# -----------------------------
DEFAULT_EXCLUDED_PREFIXES = {"APO"}  # add more in sidebar


# -----------------------------
# Utils
# -----------------------------
def normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def looks_numeric(s: str) -> bool:
    s = str(s).strip()
    if not s:
        return False
    s = s.replace(",", ".")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def strip_prefixes(name: str, prefixes: set) -> str:
    """
    Removes manufacturer prefixes ONLY if they are the first word.
    Example: "APO METFORMIN 500MG TAB" -> "METFORMIN 500MG TAB"
    """
    name = normalise_space(name)
    if not name:
        return name
    words = name.split()
    if words and words[0].upper() in prefixes:
        return normalise_space(" ".join(words[1:]))
    return name


# -----------------------------
# Read files
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
    for r in all_rows:
        r = r + [None] * (max_cols - len(r))
        padded.append(r)

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
# Auto-detect & aggregate (FIXED)
# -----------------------------
def detect_quantity_column(df: pd.DataFrame) -> int:
    """
    Chooses the column with the highest ratio of numeric-ish values.
    """
    best_idx = 0
    best_score = -1

    for i in range(df.shape[1]):
        col = df.iloc[:, i].dropna().astype(str).head(300)
        if len(col) == 0:
            continue
        numeric_hits = sum(1 for v in col if to_float_or_none(v) is not None)
        score = numeric_hits / max(1, len(col))
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def detect_text_columns(df: pd.DataFrame, qty_idx: int) -> List[int]:
    """
    Picks columns likely to be description columns: non-numeric, longer text.
    """
    candidates = []
    for i in range(df.shape[1]):
        if i == qty_idx:
            continue
        sample = df.iloc[:, i].dropna().astype(str).head(200)
        if len(sample) == 0:
            continue
        # Textiness: average length * fraction non-numeric
        avg_len = sum(len(v.strip()) for v in sample) / max(1, len(sample))
        non_numeric_frac = sum(1 for v in sample if not looks_numeric(v)) / max(1, len(sample))
        score = avg_len * non_numeric_frac
        candidates.append((score, i))

    candidates.sort(reverse=True)
    # keep top 3-5 text columns (usually description is spread across a few)
    return [i for _, i in candidates[:5]]


def build_description(row: pd.Series, text_cols: List[int]) -> str:
    parts = []
    for i in text_cols:
        v = row.iloc[i]
        if v is None:
            continue
        s = normalise_space(v)
        if not s:
            continue
        # ignore cells that are just separators or tiny garbage
        if len(s) <= 1:
            continue
        parts.append(s)
    return normalise_space(" ".join(parts))


def aggregate_inventory_table_style(df: pd.DataFrame, prefixes: set) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Aggregates quantities by cleaned medication description, for table-style exports.
    """
    qty_idx = detect_quantity_column(df)
    text_cols = detect_text_columns(df, qty_idx)

    totals = {}
    parsed = 0
    skipped = 0

    for _, row in df.iterrows():
        qty = to_float_or_none(row.iloc[qty_idx])
        if qty is None:
            skipped += 1
            continue

        desc = build_description(row, text_cols)
        desc = strip_prefixes(desc, prefixes)

        # Extra cleanup: drop pure units-only rows like "0.4 mL" if they happen
        # Keep if there's at least one letter AND at least 6 chars
        if not re.search(r"[A-Za-zÀ-ÿ]", desc) or len(desc) < 6:
            skipped += 1
            continue

        totals[desc] = totals.get(desc, 0.0) + float(qty)
        parsed += 1

    out = (
        pd.DataFrame(list(totals.items()), columns=["Name", "Amount"])
        .sort_values("Name")
        .reset_index(drop=True)
    )

    return out, qty_idx, parsed, skipped


# -----------------------------
# PDF export (optional)
# -----------------------------
def dataframe_to_simple_pdf(df: pd.DataFrame, title: str) -> bytes:
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
    lh = 12

    c.drawString(x, y, f"{'Name':70} {'Amount':>10}")
    y -= lh
    c.drawString(x, y, "-" * 90)
    y -= lh

    for _, r in df.iterrows():
        name = str(r["Name"])[:70]
        amt = f"{r['Amount']:.2f}"
        line = f"{name:70} {amt:>10}"

        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 10)

        c.drawString(x, y, line)
        y -= lh

    c.save()
    return buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Inventaire — Nettoyage & Fusion", layout="wide")
st.title("Inventaire réorganisé (prefixes supprimés + totaux)")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Prefixes à ignorer")
    excluded_text = st.text_input(
        "Séparés par virgules",
        value=",".join(sorted(DEFAULT_EXCLUDED_PREFIXES)),
        help="Ex: APO,SANDOZ,TEVA,MYLAN"
    )
    prefixes = {p.strip().upper() for p in excluded_text.split(",") if p.strip()}

    st.divider()
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
st.subheader("Preview")
st.dataframe(df_raw.head(40), use_container_width=True)

out, qty_idx, parsed, skipped = aggregate_inventory_table_style(df_raw, prefixes)

st.caption(f"Auto-detected quantity column: {qty_idx}")
st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

st.subheader("Résultat")
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
        pdf_bytes = dataframe_to_simple_pdf(out, "Inventaire réorganisé (prefixes supprimés + totaux)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_reorganise.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
