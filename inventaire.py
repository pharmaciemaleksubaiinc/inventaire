# app.py
# Streamlit Med Logger — supports CSV / Excel / PDF (via pdfplumber)
# Run:
#   pip install streamlit pandas openpyxl pdfplumber
#   streamlit run app.py

import io
import re
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import pdfplumber


# -----------------------------
# Config
# -----------------------------
DEFAULT_EXCLUDED_WORDS = {"APO"}  # add more: SANDOZ, TEVA, MYLAN, etc.


# -----------------------------
# Parsing helpers (your original logic, but safer)
# -----------------------------
def normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_med_cell(cell: object) -> Optional[Tuple[str, str]]:
    """
    Matches your original rules:
      - treat the "med info" cell as multi-line text
      - skip row if first char isn't a digit
      - lines[1] is medName
      - lines[3].split()[0] is manufacturer prefix (if exists)
    Returns (med_name, manufacturer_prefix) or None if not a medication row.
    """
    if cell is None:
        return None

    entry = str(cell).strip()
    if not entry:
        return None

    # Original condition: entry[0].isdecimal()
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
    """
    Your original:
      float(entry2.splitlines()[2].split()[0])
    but with guardrails + fallback.
    """
    if cell is None:
        return None

    text = str(cell).strip()
    if not text:
        return None

    lines = text.splitlines()

    # Primary: 3rd line, first token (like your code)
    if len(lines) >= 3:
        parts = lines[2].split()
        if parts:
            token = parts[0].replace(",", ".")
            try:
                return float(token)
            except Exception:
                pass

    # Fallback: first number anywhere
    m = re.search(r"(-?\d+(?:[.,]\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    return None


def aggregate_inventory(df: pd.DataFrame, med_col_idx: int, amt_col_idx: int, excluded_words: set):
    """
    Aggregates medication amounts by name.
    Keeps your manufacturer-prefix stripping logic:
      if medName first word == manufacturer prefix OR first word in excluded_words => remove it
    """
    hashmap = {}
    parsed = 0
    skipped = 0

    # Like your code: start from row index 1 (skip first row)
    med_col = df.iloc[1:, med_col_idx]
    amt_col = df.iloc[1:, amt_col_idx]

    for med_cell, amt_cell in zip(med_col, amt_col):
        parsed_med = parse_med_cell(med_cell)
        if not parsed_med:
            skipped += 1
            continue

        med_name, manufacturer_prefix = parsed_med
        amount = parse_amount(amt_cell)

        if amount is None:
            skipped += 1
            continue

        words = med_name.split()
        first_word = words[0].upper() if words else ""

        if first_word and (first_word == manufacturer_prefix.upper() or first_word in excluded_words):
            med_name = " ".join(words[1:]).strip()

        med_name = normalise_whitespace(med_name)
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
# File reading (CSV / Excel / PDF via pdfplumber)
# -----------------------------
def read_pdf_to_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Extracts tables from PDF pages using pdfplumber.
    Returns a dataframe built from concatenated table rows.

    Notes:
    - Works best when the PDF has real tables (not scanned images).
    - pdfplumber outputs list-of-rows, each row is list of cell strings (or None).
    """
    all_rows = []
    max_cols = 0

    # pdfplumber can open a file-like object, but Streamlit's UploadedFile is fine.
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row is None:
                        continue
                    max_cols = max(max_cols, len(row))
                    all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No tables detected in PDF. If this PDF is a scan/image, you need OCR instead.")

    # Pad rows to same width
    padded = []
    for row in all_rows:
        row = list(row)
        row += [None] * (max_cols - len(row))
        padded.append(row)

    df = pd.DataFrame(padded)
    return df


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, "Loaded CSV"

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df, "Loaded Excel"

    if name.endswith(".pdf"):
        df = read_pdf_to_dataframe(uploaded_file)
        return df, "Loaded PDF (via pdfplumber)"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, or PDF.")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Med Logger (Inventory Aggregator)", layout="wide")
st.title("Med Logger — Inventory Aggregator (CSV / Excel / PDF)")

st.write(
    "Upload your inventory file, choose the columns for **Medication info** and **Amount**, "
    "then it will merge manufacturer variants (like APO) and sum totals."
)

uploaded = st.file_uploader("Upload inventory file", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Settings")
    excluded_text = st.text_input(
        "Prefixes to exclude (comma-separated)",
        value=",".join(sorted(DEFAULT_EXCLUDED_WORDS)),
        help="Example: APO,SANDOZ,TEVA"
    )
    excluded_words = {w.strip().upper() for w in excluded_text.split(",") if w.strip()}

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw, note = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(note)

st.subheader("Preview of uploaded data")
st.dataframe(df_raw.head(40), use_container_width=True)

st.subheader("Pick columns")
col_count = df_raw.shape[1]

default_med_col = 1 if col_count > 1 else 0
default_amt_col = 2 if col_count > 2 else min(1, col_count - 1)

med_col_idx = st.number_input(
    "Medication info column index (0-based)",
    min_value=0,
    max_value=max(0, col_count - 1),
    value=default_med_col,
    step=1,
)

amt_col_idx = st.number_input(
    "Amount column index (0-based)",
    min_value=0,
    max_value=max(0, col_count - 1),
    value=default_amt_col,
    step=1,
)

if med_col_idx == amt_col_idx:
    st.warning("Medication column and Amount column are the same. Pick two different columns.")

run = st.button("Aggregate", type="primary")

if run:
    try:
        out, parsed, skipped = aggregate_inventory(
            df_raw,
            int(med_col_idx),
            int(amt_col_idx),
            excluded_words
        )
    except Exception as e:
        st.error(f"Aggregation failed: {e}")
        st.stop()

    st.subheader("Result")
    st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")
    st.dataframe(out, use_container_width=True)

    st.subheader("Download")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="med_aggregated.csv",
        mime="text/csv",
    )

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Aggregated")

    st.download_button(
        "Download Excel",
        data=excel_buf.getvalue(),
        file_name="med_aggregated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
