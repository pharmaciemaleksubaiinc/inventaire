import io
import re
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# For PDF table extraction
try:
    import camelot  # pip install camelot-py[cv]
    CAMELOT_OK = True
except Exception:
    CAMELOT_OK = False

# -----------------------------
# Config
# -----------------------------
DEFAULT_EXCLUDED_WORDS = {"APO"}  # add more if needed


# -----------------------------
# Helpers
# -----------------------------
def normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_amount(cell: str) -> Optional[float]:
    """
    Tries to replicate your:
      float(entry2.splitlines()[2].split()[0])
    but with safeguards + fallback parsing.
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None

    lines = text.splitlines()

    # Primary attempt: exactly like your original code (3rd line, first token)
    if len(lines) >= 3:
        token = lines[2].split()[0] if lines[2].split() else ""
        token = token.replace(",", ".")
        try:
            return float(token)
        except Exception:
            pass

    # Fallback: find first number anywhere
    m = re.search(r"(-?\d+(?:[.,]\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    return None


def parse_med_cell(cell: str) -> Optional[Tuple[str, str]]:
    """
    Replicates your parsing of the medication info cell:
      - skip if first char isn't a digit
      - lines[1] -> medName
      - lines[3].split()[0] -> manufacturer prefix (if present)

    Returns (med_name, manufacturer_prefix) or None if not a medication row.
    """
    if cell is None:
        return None

    entry = str(cell).strip()
    if not entry:
        return None

    # Skip non-medication rows
    first_char = entry[0]
    if not first_char.isdecimal():
        return None

    lines = entry.splitlines()

    # Need at least 2 lines for lines[1]
    if len(lines) < 2:
        return None

    med_name = lines[1].strip()

    manufacturer_prefix = ""
    if len(lines) >= 4 and lines[3].strip():
        manufacturer_prefix = lines[3].split()[0].strip()

    return med_name, manufacturer_prefix


def aggregate_inventory(df: pd.DataFrame, med_col_idx: int, amt_col_idx: int, excluded_words: set) -> pd.DataFrame:
    """
    Builds a dataframe with columns: Name, Amount (summed).
    """
    hashmap = {}

    # Your original started at row 1 (skip first row)
    med_col = df.iloc[1:, med_col_idx]
    amt_col = df.iloc[1:, amt_col_idx]

    skipped = 0
    parsed = 0

    for med_cell, amt_cell in zip(med_col, amt_col):
        med_parsed = parse_med_cell(med_cell)
        if not med_parsed:
            skipped += 1
            continue

        med_name, manufacturer_prefix = med_parsed
        amount = parse_amount(amt_cell)

        if amount is None:
            skipped += 1
            continue

        # Remove prefix if the first word equals manufacturer prefix OR is in excluded list
        med_words = med_name.split()
        first_word = med_words[0] if med_words else ""

        if first_word and (first_word == manufacturer_prefix or first_word in excluded_words):
            med_name = " ".join(med_words[1:]).strip()

        med_name = normalise_whitespace(med_name)
        if not med_name:
            skipped += 1
            continue

        hashmap[med_name] = hashmap.get(med_name, 0.0) + float(amount)
        parsed += 1

    out = (
        pd.DataFrame(list(hashmap.items()), columns=["Name", "Amount"])
        .sort_values(["Name"], ascending=True)
        .reset_index(drop=True)
    )

    return out, parsed, skipped


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, source_note). For PDF, returns the first detected table by default.
    """
    name = uploaded_file.name.lower()

    # CSV
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, "Loaded CSV"

    # Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df, "Loaded Excel"

    # PDF
    if name.endswith(".pdf"):
        if not CAMELOT_OK:
            raise RuntimeError("PDF support requires camelot. Install: pip install camelot-py[cv]")

        # Camelot needs a filepath; write to a temp buffer file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Try lattice first (best for ruled tables), then stream
        tables = []
        try:
            tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
        except Exception:
            pass
        if not tables:
            tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")

        if not tables or len(tables) == 0:
            raise RuntimeError("No tables detected in the PDF.")

        # Pick the biggest table (most rows * cols)
        best = max(tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
        df = best.df

        # Camelot outputs everything as strings, and often includes header-like first row
        return df, f"Loaded PDF table (pages=all, picked largest table {df.shape[0]}x{df.shape[1]})"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, or PDF.")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Med Logger (Inventory Aggregator)", layout="wide")
st.title("Med Logger — Inventory Aggregator")

st.write("Upload **CSV / Excel / PDF**. Then pick the columns that contain the **med cell** and the **amount cell**.")

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

# Show preview
st.subheader("Preview of uploaded data")
st.dataframe(df_raw.head(30), use_container_width=True)

# Column picking
st.subheader("Pick columns")
col_count = df_raw.shape[1]
default_med_col = 1 if col_count > 1 else 0
default_amt_col = 2 if col_count > 2 else min(1, col_count - 1)

med_col_idx = st.number_input("Medication info column index (0-based)", min_value=0, max_value=col_count - 1, value=default_med_col)
amt_col_idx = st.number_input("Amount column index (0-based)", min_value=0, max_value=col_count - 1, value=default_amt_col)

if med_col_idx == amt_col_idx:
    st.warning("Medication column and Amount column are the same. Pick two different columns.")

run = st.button("Aggregate")

if run:
    try:
        out, parsed, skipped = aggregate_inventory(df_raw, int(med_col_idx), int(amt_col_idx), excluded_words)
    except Exception as e:
        st.error(f"Aggregation failed: {e}")
        st.stop()

    st.subheader("Result")
    st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")
    st.dataframe(out, use_container_width=True)

    # Downloads
    st.subheader("Download")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="med_aggregated.csv", mime="text/csv")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Aggregated")
    st.download_button(
        "Download Excel",
        data=excel_buf.getvalue(),
        file_name="med_aggregated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
