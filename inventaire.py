# inventaire.py
# Install:
#   pip install streamlit pandas openpyxl pdfplumber pdfminer.six
# Optional PDF export:
#   pip install reportlab
#
# Run:
#   streamlit run inventaire.py

import io
import re
from typing import List, Optional, Tuple, Dict, Set

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
# Utils
# -----------------------------
def normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


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
            for table in (page.extract_tables() or []):
                for row in (table or []):
                    if row is None:
                        continue
                    row = list(row)
                    max_cols = max(max_cols, len(row))
                    all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No tables detected in PDF. If it's a scanned PDF, you need OCR.")

    padded = [r + [None] * (max_cols - len(r)) for r in all_rows]
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
# Auto-detect columns (robust)
# -----------------------------
def detect_quantity_candidates(df: pd.DataFrame, sample_n: int = 600) -> List[Tuple[float, int]]:
    """
    Scores each column by how numeric it is.
    Returns list of (score, idx) sorted desc.
    """
    scored = []
    for i in range(df.shape[1]):
        col = df.iloc[:, i].dropna().head(sample_n)
        if len(col) == 0:
            scored.append((0.0, i))
            continue

        numeric = [to_float_or_none(v) for v in col]
        hits = sum(1 for v in numeric if v is not None)
        score = hits / max(1, len(col))
        scored.append((score, i))

    scored.sort(reverse=True)
    return scored


def detect_text_columns(df: pd.DataFrame, qty_idx: int) -> List[int]:
    """
    Picks columns likely to be description columns (texty).
    """
    candidates = []
    for i in range(df.shape[1]):
        if i == qty_idx:
            continue
        sample = df.iloc[:, i].dropna().astype(str).head(350)
        if len(sample) == 0:
            continue
        avg_len = sum(len(s.strip()) for s in sample) / max(1, len(sample))
        # favour columns that are mostly non-numeric
        non_num = sum(1 for s in sample if to_float_or_none(s) is None) / max(1, len(sample))
        candidates.append((avg_len * non_num, i))

    candidates.sort(reverse=True)
    return [i for _, i in candidates[:6]]


def build_description(row: pd.Series, text_cols: List[int]) -> str:
    parts = []
    for i in text_cols:
        v = row.iloc[i]
        if v is None:
            continue
        s = normalise_space(v)
        if not s or len(s) <= 1:
            continue
        parts.append(s)
    return normalise_space(" ".join(parts))


# -----------------------------
# Prefix logic (safe + “combine regardless of prefix”)
# -----------------------------
def learn_prefixes_from_hyphens(descriptions: pd.Series) -> Set[str]:
    """
    Learn prefixes ONLY from hyphenated forms found in the file:
      AA-FLUOXETINE -> AA
    This is safe and requires no “min count” slider.
    """
    prefixes = set()
    for raw in descriptions.dropna().astype(str).head(15000):
        s = normalise_space(raw)
        m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
        if m:
            p = m.group(1).upper()
            if p.isalpha():
                prefixes.add(p)
    return prefixes


def split_prefix_and_core(desc: str, hyphen_prefixes: Set[str]) -> Tuple[Optional[str], str]:
    """
    Combine meds regardless of prefix:
      - If it's hyphen-prefix: strip always.
      - If it's space-prefix: strip only if that token is known from hyphen-prefixes in THIS file.
        (prevents stripping manufacturer names like ASTELLAS/BAUSCH/etc.)
    """
    s = normalise_space(desc)
    if not s:
        return None, s

    # Hyphen form: always strip
    m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
    if m:
        p = m.group(1).upper()
        rest = normalise_space(m.group(2))
        if p.isalpha() and rest:
            return p, rest

    # Space form: strip only if that prefix was seen in hyphen form somewhere in the file
    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in hyphen_prefixes:
            return first, normalise_space(" ".join(parts[1:]))

    return None, s


def is_reasonable_name(core: str) -> bool:
    """
    Avoid grouping junk rows like "0.4 mL" or "Plaquette".
    Needs letters and at least 6 chars.
    """
    c = normalise_space(core)
    return bool(re.search(r"[A-Za-zÀ-ÿ]", c)) and len(c) >= 6


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_inventory(df: pd.DataFrame, qty_idx: int) -> Tuple[pd.DataFrame, int, int, Set[str]]:
    text_cols = detect_text_columns(df, qty_idx)

    # Build descriptions to learn hyphen-prefixes from the actual file
    descs = []
    for _, row in df.iterrows():
        d = build_description(row, text_cols)
        if d:
            descs.append(d)
    hyphen_prefixes = learn_prefixes_from_hyphens(pd.Series(descs))

    totals: Dict[str, float] = {}
    brands: Dict[str, Set[str]] = {}

    parsed = 0
    skipped = 0

    for _, row in df.iterrows():
        qty = to_float_or_none(row.iloc[qty_idx])
        if qty is None:
            skipped += 1
            continue

        desc = build_description(row, text_cols)
        if not desc:
            skipped += 1
            continue

        prefix, core = split_prefix_and_core(desc, hyphen_prefixes)

        if not is_reasonable_name(core):
            skipped += 1
            continue

        totals[core] = totals.get(core, 0.0) + float(qty)
        if prefix:
            brands.setdefault(core, set()).add(prefix)

        parsed += 1

    out = pd.DataFrame(
        {
            "Médicament (regroupé)": list(totals.keys()),
            "Quantité totale": [totals[k] for k in totals.keys()],
            "Préfixes / Marques trouvés": [", ".join(sorted(brands.get(k, set()))) for k in totals.keys()],
        }
    ).sort_values("Médicament (regroupé)").reset_index(drop=True)

    out["Préfixes / Marques trouvés"] = out["Préfixes / Marques trouvés"].fillna("")
    return out, parsed, skipped, hyphen_prefixes


# -----------------------------
# Optional PDF export
# -----------------------------
def dataframe_to_simple_pdf(df: pd.DataFrame, title: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("PDF export requires reportlab. Add 'reportlab' to requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    x = 0.65 * inch
    y = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 9)
    lh = 11

    headers = ["Médicament (regroupé)", "Quantité totale", "Préfixes / Marques trouvés"]
    c.drawString(x, y, f"{headers[0][:55]:55}  {headers[1]:>12}  {headers[2]}")
    y -= lh
    c.drawString(x, y, "-" * 115)
    y -= lh

    for _, r in df.iterrows():
        name = str(r[headers[0]])[:55]
        qty = f"{float(r[headers[1]]):.2f}"
        pref = str(r[headers[2]])[:45]
        line = f"{name:55}  {qty:>12}  {pref}"

        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 9)

        c.drawString(x, y, line)
        y -= lh

    c.save()
    return buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Inventaire — Regroupement", layout="wide")
st.title("Inventaire — Regroupement des médicaments (préfixes fusionnés)")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
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

# Quantity detection with fallback UI
candidates = detect_quantity_candidates(df_raw)
best_score, best_idx = candidates[0]

st.caption(f"Auto quantity guess: column {best_idx} (numeric score {best_score:.2f})")

qty_idx = best_idx
if best_score < 0.30:  # weak detection -> force human choice
    st.warning("Quantity column auto-detection is weak for this file. Pick the right quantity column below.")
    options = [f"Col {i} (score {s:.2f})" for s, i in candidates[: min(10, len(candidates))]]
    choice = st.selectbox("Select quantity column", options, index=0)
    qty_idx = int(re.search(r"Col (\d+)", choice).group(1))

out, parsed, skipped, hyphen_prefixes = aggregate_inventory(df_raw, qty_idx)

st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

with st.expander("Préfixes hyphen détectés (debug)"):
    st.write(", ".join(sorted(hyphen_prefixes))[:4000])

st.subheader("Résultat (regroupé)")
st.dataframe(out, use_container_width=True)

# Excel download
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Inventaire_regroupe")

st.download_button(
    "Télécharger Excel (.xlsx)",
    data=excel_buf.getvalue(),
    file_name="inventaire_regroupe.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Optional PDF download
if export_pdf:
    try:
        pdf_bytes = dataframe_to_simple_pdf(out, "Inventaire regroupé (préfixes fusionnés)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_regroupe.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
