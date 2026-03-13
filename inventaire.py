# inventaire.py
# Simple version:
# - Upload CSV / Excel / PDF
# - Detect quantity column
# - Clean description
# - Remove vendors
# - Remove generic prefixes at the start
# - Extract molecule
# - Sum stock quantity by molecule
# - Also show probable unit/form if detectable
#
# Install:
#   pip install streamlit pandas openpyxl pdfplumber pdfminer.six
# Optional PDF export:
#   pip install reportlab
#
# Run:
#   streamlit run inventaire.py

import io
import re
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Dict, Set

import pandas as pd
import streamlit as st

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except Exception:
    PDFPLUMBER_OK = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_VENDORS = [
    "McKesson",
    "Pharma Plus",
    "PharmaPlus",
    "Pharma+",
]

# These are generic/company prefixes to ignore ONLY at the beginning of the med name
DEFAULT_PREFIXES = [
    "APO", "PMS", "RATIO", "SANDOZ", "TEVA", "AURO", "JAMP", "MINT",
    "MYLAN", "TARO", "ACT", "NOVO", "BIO", "OPUS", "RAN", "MAR",
    "AA", "ACC", "ACH", "AG",
]

# Common non-molecule tokens
STOPWORDS = {
    "PHARMA", "PHARMACEUTICAL", "PHARMACEUTICALS", "PHARMACEUTICA",
    "LAB", "LABS", "LABORATORIES", "INC", "LTD", "LIMITED", "CORP",
    "CORPORATION", "CANADA", "HEALTH", "TRADING", "GROUP", "COMPANY",
    "MCKESSON", "PHARMAPLUS", "PHARMA+", "PLUS",
    "TAB", "TABS", "TABLET", "TABLETS",
    "CAP", "CAPS", "CAPSULE", "CAPSULES",
    "COMP", "COMPRIME", "COMPRIMES", "COMPRIMÉ", "COMPRIMÉS",
    "SUSP", "SUSPENSION", "SOLUTION", "SYRUP", "SIROP",
    "CREME", "CRÈME", "POMMADE", "GEL", "LOTION",
    "INJ", "INJECTION", "VIAL", "AMPOULE", "PATCH", "SPRAY",
    "DROPS", "GOUTTES", "XR", "SR", "ER", "CR", "DR",
    "MG", "MCG", "G", "KG", "ML", "L", "IU", "UNIT", "UNITS",
    "BOTTLE", "BOTTLES", "BOX", "BOXES", "PACK", "PACKS",
    "PLAQUETTE", "BANDELETTE",
}

UNIT_MAP = {
    "TAB": "tablets",
    "TABS": "tablets",
    "TABLET": "tablets",
    "TABLETS": "tablets",
    "COMP": "tablets",
    "COMPRIME": "tablets",
    "COMPRIMES": "tablets",
    "COMPRIMÉ": "tablets",
    "COMPRIMÉS": "tablets",

    "CAP": "capsules",
    "CAPS": "capsules",
    "CAPSULE": "capsules",
    "CAPSULES": "capsules",

    "BOTTLE": "bottles",
    "BOTTLES": "bottles",
    "VIAL": "vials",
    "VIALS": "vials",
    "BOX": "boxes",
    "BOXES": "boxes",
    "PACK": "packs",
    "PACKS": "packs",

    "ML": "mL",
    "L": "L",
    "G": "g",
    "MG": "mg",
    "MCG": "mcg",
}


# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
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


# -----------------------------
# File reading
# -----------------------------
def read_pdf_to_df(uploaded_file) -> pd.DataFrame:
    if not PDFPLUMBER_OK:
        raise RuntimeError("PDF support not installed. Add pdfplumber to requirements.txt.")

    rows: List[List[object]] = []
    max_cols = 0

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                for row in (table or []):
                    if row is None:
                        continue
                    row = list(row)
                    max_cols = max(max_cols, len(row))
                    rows.append(row)

    if not rows:
        raise RuntimeError("No tables detected in PDF. If the PDF is scanned, you need OCR.")

    padded = [r + [None] * (max_cols - len(r)) for r in rows]
    return pd.DataFrame(padded)


def read_uploaded(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), "Loaded CSV"
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file), "Loaded Excel"
    if name.endswith(".pdf"):
        return read_pdf_to_df(uploaded_file), "Loaded PDF (via pdfplumber)"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, or PDF.")


# -----------------------------
# Column detection
# -----------------------------
def detect_quantity_candidates(df: pd.DataFrame, sample_n: int = 800) -> List[Tuple[float, int]]:
    scored = []
    for i in range(df.shape[1]):
        col = df.iloc[:, i].dropna().head(sample_n)
        if len(col) == 0:
            scored.append((0.0, i))
            continue
        hits = sum(1 for v in col if to_float_or_none(v) is not None)
        scored.append((hits / max(1, len(col)), i))
    scored.sort(reverse=True)
    return scored


def detect_text_columns(df: pd.DataFrame, qty_idx: int) -> List[int]:
    candidates = []
    for i in range(df.shape[1]):
        if i == qty_idx:
            continue
        sample = df.iloc[:, i].dropna().astype(str).head(400)
        if len(sample) == 0:
            continue
        avg_len = sum(len(s.strip()) for s in sample) / max(1, len(sample))
        non_num = sum(1 for s in sample if not looks_numeric_cell(s)) / max(1, len(sample))
        candidates.append((avg_len * non_num, i))
    candidates.sort(reverse=True)
    return [i for _, i in candidates[:6]]


def build_description(row: pd.Series, cols: List[int]) -> str:
    parts = []
    for i in cols:
        v = row.iloc[i]
        if v is None:
            continue
        s = norm(v)
        if s and len(s) > 1:
            parts.append(s)
    return norm(" ".join(parts))


# -----------------------------
# Cleaning logic
# -----------------------------
def remove_vendors_and_manufacturer(text: str, vendors: List[str]) -> Tuple[str, Set[str]]:
    """
    If a vendor appears in the string, everything before the first vendor is treated
    as manufacturer/company clutter and removed.
    Vendor words themselves are also removed from the medication description, but kept separately.
    """
    s = norm(text)
    found = set()
    if not s:
        return s, found

    earliest = None
    chosen_vendor = None

    for v in sorted({norm(x) for x in vendors if norm(x)}, key=len, reverse=True):
        m = re.search(rf"(?i)\b{re.escape(v)}\b", s)
        if m:
            found.add(v.upper())
            if earliest is None or m.start() < earliest:
                earliest = m.start()
                chosen_vendor = v

    if earliest is not None and chosen_vendor is not None:
        s = norm(s[earliest + len(chosen_vendor):])

    for v in sorted({norm(x) for x in vendors if norm(x)}, key=len, reverse=True):
        if re.search(rf"(?i)\b{re.escape(v)}\b", s):
            found.add(v.upper())
            s = re.sub(rf"(?i)\b{re.escape(v)}\b", " ", s)

    return norm(s), found


def strip_generic_prefix(text: str, prefixes: Set[str]) -> Tuple[Optional[str], str]:
    """
    Remove prefix only if it appears at the start:
      APO-FLUOXETINE -> FLUOXETINE
      APO FLUOXETINE -> FLUOXETINE
    """
    s = norm(text)
    if not s:
        return None, s

    m = re.match(r"^([A-Za-z]{1,12})\s*-\s*(.+)$", s)
    if m:
        p = m.group(1).upper()
        rest = norm(m.group(2))
        if p in prefixes and rest:
            return p, rest

    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in prefixes:
            return first, norm(" ".join(parts[1:]))

    return None, s


def detect_unit(text: str) -> str:
    s = norm(text).upper()

    for raw, canonical in UNIT_MAP.items():
        if re.search(rf"\b{re.escape(raw)}\b", s):
            return canonical

    return "units"


def extract_molecule(text: str) -> str:
    """
    Goal: get the actual molecule, not company/vendor/form noise.
    We choose the first decent token after cleaning.
    """
    s = norm(text).upper()
    if not s:
        return ""

    tokens = [t.strip(" ,;()[]{}:+-/") for t in s.split()]

    for tok in tokens:
        if not tok:
            continue
        if tok in STOPWORDS:
            continue
        if re.search(r"\d", tok):
            continue
        if len(tok) < 4:
            continue
        if not re.search(r"[A-ZÀ-Ý]", tok):
            continue
        return tok

    return ""


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_inventory(df: pd.DataFrame, qty_idx: int, vendors: List[str], prefixes: Set[str]):
    text_cols = detect_text_columns(df, qty_idx)

    totals: Dict[str, float] = defaultdict(float)
    unit_counts: Dict[str, Counter] = defaultdict(Counter)
    prefix_seen: Dict[str, Set[str]] = defaultdict(set)
    vendor_seen: Dict[str, Set[str]] = defaultdict(set)
    sample_desc: Dict[str, str] = {}

    parsed = 0
    skipped = 0

    for _, row in df.iterrows():
        qty = to_float_or_none(row.iloc[qty_idx])
        if qty is None:
            skipped += 1
            continue

        full_desc = build_description(row, text_cols)
        if not full_desc:
            skipped += 1
            continue

        cleaned_desc, vendors_found = remove_vendors_and_manufacturer(full_desc, vendors)
        prefix_found, core_desc = strip_generic_prefix(cleaned_desc, prefixes)
        molecule = extract_molecule(core_desc)
        if not molecule:
            skipped += 1
            continue

        probable_unit = detect_unit(core_desc)

        totals[molecule] += float(qty)
        unit_counts[molecule][probable_unit] += 1

        if prefix_found:
            prefix_seen[molecule].add(prefix_found)
        if vendors_found:
            vendor_seen[molecule].update(vendors_found)
        if molecule not in sample_desc:
            sample_desc[molecule] = core_desc

        parsed += 1

    rows = []
    for molecule in sorted(totals.keys()):
        most_common_unit = unit_counts[molecule].most_common(1)[0][0] if unit_counts[molecule] else "units"
        rows.append({
            "Molécule": molecule,
            "Quantité totale en stock": totals[molecule],
            "Unité probable": most_common_unit,
            "Exemple description": sample_desc.get(molecule, ""),
            "Préfixes trouvés": ", ".join(sorted(prefix_seen.get(molecule, set()))),
            "Vendeurs trouvés": ", ".join(sorted(vendor_seen.get(molecule, set()))),
        })

    out = pd.DataFrame(rows)
    return out, parsed, skipped


# -----------------------------
# Optional PDF export
# -----------------------------
def df_to_pdf(df: pd.DataFrame, title: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("PDF export requires reportlab. Add 'reportlab' to requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter

    x = 0.6 * inch
    y = height - 0.7 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 8)
    line_h = 10

    headers = ["Molécule", "Quantité totale en stock", "Unité probable"]
    c.drawString(x, y, f"{headers[0][:20]:20} {headers[1]:>20}  {headers[2][:14]}")
    y -= line_h
    c.drawString(x, y, "-" * 70)
    y -= line_h

    for _, row in df.iterrows():
        line = f"{str(row['Molécule'])[:20]:20} {float(row['Quantité totale en stock']):>20.2f}  {str(row['Unité probable'])[:14]}"
        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.7 * inch
            c.setFont("Helvetica", 8)
        c.drawString(x, y, line)
        y -= line_h

    c.save()
    return buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Inventaire par molécule", layout="wide")
st.title("Inventaire par molécule")
st.write("Résultat simple : combien d’unités tu as en stock pour chaque molécule.")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Vendeurs à retirer du nom")
    vendors_text = st.text_area("Une entrée par ligne", value="\n".join(DEFAULT_VENDORS), height=120)
    vendors = [norm(v) for v in vendors_text.splitlines() if norm(v)]

    st.divider()
    st.header("Préfixes génériques à ignorer")
    prefixes_text = st.text_area("Une entrée par ligne", value="\n".join(DEFAULT_PREFIXES), height=220)
    prefixes = {norm(p).upper() for p in prefixes_text.splitlines() if norm(p)}

    st.divider()
    show_debug = st.checkbox("Afficher colonnes debug", value=False)
    export_pdf = st.checkbox("Générer aussi un PDF", value=False)

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw, note = read_uploaded(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(note)
st.subheader("Preview")
st.dataframe(df_raw.head(40), use_container_width=True)

candidates = detect_quantity_candidates(df_raw)
best_score, best_idx = candidates[0]
st.caption(f"Auto quantity guess: column {best_idx} (numeric score {best_score:.2f})")

qty_idx = best_idx
if best_score < 0.30:
    st.warning("Quantity auto-detection is weak. Pick the correct quantity column.")
    options = [f"Col {i} (score {s:.2f})" for s, i in candidates[:min(12, len(candidates))]]
    choice = st.selectbox("Select quantity column", options, index=0)
    qty_idx = int(re.search(r"Col (\d+)", choice).group(1))

out, parsed, skipped = aggregate_inventory(df_raw, qty_idx, vendors, prefixes)

st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

display_df = out.copy()
if not show_debug and not display_df.empty:
    keep = ["Molécule", "Quantité totale en stock", "Unité probable"]
    display_df = display_df[keep]

st.subheader("Résultat")
st.dataframe(display_df, use_container_width=True)

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Inventaire_par_molecule")

st.download_button(
    "Télécharger Excel (.xlsx)",
    data=excel_buf.getvalue(),
    file_name="inventaire_par_molecule.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if export_pdf:
    try:
        pdf_bytes = df_to_pdf(display_df, "Inventaire par molécule")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_par_molecule.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
