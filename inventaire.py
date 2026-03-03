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
# Vendor words to remove (anywhere in text)
# Add more if your export contains other suppliers/wholesalers.
# -----------------------------
DEFAULT_VENDOR_TRASH = {
    "MCKESSON", "MC KESSON", "MCK", "MCSON", "MCKESS",
    "PHARMA PLUS", "PHARMAPLUS", "PHARMA+", "PHARMA-PLUS",
}


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


def looks_numeric_cell(x) -> bool:
    return to_float_or_none(x) is not None


def remove_vendor_words(text: str, vendor_words: Set[str]) -> str:
    """
    Remove vendor tokens/phrases anywhere (whole words, case-insensitive).
    Handles multi-word phrases like "PHARMA PLUS".
    """
    s = normalise_space(text)
    if not s:
        return s

    # Sort by length so multi-word phrases removed first
    for w in sorted(vendor_words, key=len, reverse=True):
        w_norm = normalise_space(w)
        if not w_norm:
            continue
        # whole word / whole phrase boundaries
        s = re.sub(rf"(?i)\b{re.escape(w_norm)}\b", " ", s)

    return normalise_space(s)


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
# Auto-detect columns (table-style)
# -----------------------------
def detect_quantity_column(df: pd.DataFrame) -> int:
    best_idx = 0
    best_score = -1.0

    for i in range(df.shape[1]):
        col = df.iloc[:, i].dropna().head(600)
        if len(col) == 0:
            continue
        hits = sum(1 for v in col if to_float_or_none(v) is not None)
        score = hits / max(1, len(col))
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def detect_text_columns(df: pd.DataFrame, qty_idx: int) -> List[int]:
    candidates = []
    for i in range(df.shape[1]):
        if i == qty_idx:
            continue
        sample = df.iloc[:, i].dropna().astype(str).head(350)
        if len(sample) == 0:
            continue
        avg_len = sum(len(s.strip()) for s in sample) / max(1, len(sample))
        non_num = sum(1 for s in sample if not looks_numeric_cell(s)) / max(1, len(sample))
        candidates.append((avg_len * non_num, i))

    candidates.sort(reverse=True)
    return [i for _, i in candidates[:6]]  # join up to 6 text columns


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
# Prefix learning (from YOUR inventory)
# -----------------------------
def learn_prefixes_from_file(descriptions: pd.Series, min_count: int = 2) -> Set[str]:
    """
    Learns prefixes used in your dataset. Captures:
      - Hyphen prefixes: "AA-FLUOXETINE" => AA
      - Space prefixes used frequently: "PMS FLUOXETINE" => PMS
    We keep short alphabetic tokens only.
    """
    hyphen_counts: Dict[str, int] = {}
    first_token_counts: Dict[str, int] = {}

    for raw in descriptions.dropna().astype(str).head(10000):
        s = normalise_space(raw)
        if not s:
            continue

        # Hyphen prefix signal
        m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
        if m:
            p = m.group(1).upper()
            if p.isalpha():
                hyphen_counts[p] = hyphen_counts.get(p, 0) + 1

        # First token signal
        first = s.split()[0].upper()
        if 1 <= len(first) <= 10 and first.isalpha():
            first_token_counts[first] = first_token_counts.get(first, 0) + 1

    prefixes = set()

    # Always accept hyphen prefixes that appear enough
    for p, c in hyphen_counts.items():
        if c >= 1:  # even 1 is fine: hyphen form is very indicative
            prefixes.add(p)

    # Accept frequent first tokens as prefixes too (space-form like "PMS FLUOXETINE")
    for p, c in first_token_counts.items():
        if c >= min_count:
            prefixes.add(p)

    return prefixes


def split_prefix_and_core(desc: str, learned_prefixes: Set[str]) -> Tuple[Optional[str], str]:
    """
    Strip prefix in two cases:
      1) hyphen form ALWAYS: "AA-FLUOXETINE" -> prefix AA, core FLUOXETINE
      2) space form ONLY if prefix is learned: "PMS FLUOXETINE" -> prefix PMS, core FLUOXETINE
    """
    s = normalise_space(desc)
    if not s:
        return None, s

    # Hyphen form: always strip if it looks like a prefix token (letters, short)
    m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
    if m:
        p = m.group(1).upper()
        rest = normalise_space(m.group(2))
        if p.isalpha() and rest:
            return p, rest

    # Space form: strip only if learned
    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in learned_prefixes:
            return first, normalise_space(" ".join(parts[1:]))

    return None, s


def is_real_med_name(text: str) -> bool:
    """
    Filters out garbage like "0.4 mL", "1500 mL", "Plaquette", etc.
    Heuristic: must contain letters AND at least one of:
      - length >= 8, or
      - contains a digit AND letters (often strength/form)
    """
    s = normalise_space(text)
    if not s:
        return False

    has_letters = bool(re.search(r"[A-Za-zÀ-ÿ]", s))
    if not has_letters:
        return False

    if len(s) >= 8:
        return True

    has_digit = bool(re.search(r"\d", s))
    return has_digit


# -----------------------------
# Aggregation (core + totals + brand list)
# -----------------------------
def aggregate_inventory(df: pd.DataFrame, vendor_words: Set[str], prefix_min_count: int) -> Tuple[pd.DataFrame, int, int, int, Set[str]]:
    qty_idx = detect_quantity_column(df)
    text_cols = detect_text_columns(df, qty_idx)

    # Build descriptions to learn prefixes from the actual file
    desc_list = []
    for _, row in df.iterrows():
        d = build_description(row, text_cols)
        d = remove_vendor_words(d, vendor_words)
        if d:
            desc_list.append(d)

    learned_prefixes = learn_prefixes_from_file(pd.Series(desc_list), min_count=prefix_min_count)

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
        desc = remove_vendor_words(desc, vendor_words)

        if not desc:
            skipped += 1
            continue

        prefix, core = split_prefix_and_core(desc, learned_prefixes)

        if not is_real_med_name(core):
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

    return out, qty_idx, parsed, skipped, learned_prefixes


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
st.set_page_config(page_title="Inventaire — Regroupement (préfixes)", layout="wide")
st.title("Inventaire — Regroupement (tous préfixes) + vendors supprimés")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Vendors à supprimer (McKesson, Pharma Plus, etc.)")
    vendor_text = st.text_area(
        "Une entrée par ligne",
        value="\n".join(sorted(DEFAULT_VENDOR_TRASH)),
        height=140,
    )
    vendor_words = {normalise_space(v).upper() for v in vendor_text.splitlines() if normalise_space(v)}

    st.divider()
    st.header("Détection des préfixes")
    prefix_min_count = st.slider(
        "Un token doit apparaître au moins N fois pour être traité comme préfixe (space-form)",
        min_value=1,
        max_value=20,
        value=2,
        help="Hyphen-form (AA-XXX) is always stripped. Space-form (PMS XXX) needs frequency to avoid false positives.",
    )

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

out, qty_idx, parsed, skipped, learned_prefixes = aggregate_inventory(df_raw, vendor_words, prefix_min_count)

st.caption(f"Auto-detected quantity column: {qty_idx}")
st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

with st.expander("Préfixes détectés (debug)"):
    st.write(", ".join(sorted(learned_prefixes))[:4000])

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
        pdf_bytes = dataframe_to_simple_pdf(out, "Inventaire regroupé (vendors supprimés, préfixes fusionnés)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_regroupe.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
