# inventaire.py
# Simple and safer version:
# - Upload CSV / Excel / PDF
# - YOU choose the stock quantity column manually
# - App builds a medication description from text columns
# - Removes vendor names from the description
# - Ignores generic prefixes at the start (APO, PMS, AURO, etc.)
# - Extracts the molecule name
# - Sums the stock quantity by molecule
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
from collections import defaultdict, Counter
from typing import List, Optional, Tuple, Dict, Set

import pandas as pd
import streamlit as st

# Optional PDF reading
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
# Defaults
# -----------------------------
DEFAULT_VENDORS = [
    "McKesson",
    "Pharma Plus",
    "PharmaPlus",
    "Pharma+",
]

# Prefixes to ignore at the START of the drug name
DEFAULT_PREFIXES = [
    "APO", "PMS", "RATIO", "SANDOZ", "TEVA", "AURO", "JAMP", "MINT",
    "MYLAN", "TARO", "ACT", "NOVO", "BIO", "OPUS", "RAN", "MAR",
    "AA", "ACC", "ACH", "AG", "BGP",
]

# Words we do NOT want to become molecules
STOPWORDS = {
    # company / vendor / generic filler
    "PHARMA", "PHARMACEUTICAL", "PHARMACEUTICALS", "PHARMACEUTICA",
    "LAB", "LABS", "LABORATORIES", "INC", "LTD", "LIMITED", "CORP",
    "CORPORATION", "CANADA", "HEALTH", "TRADING", "GROUP", "COMPANY",
    "MCKESSON", "PHARMAPLUS", "PLUS",

    # forms / packaging / units
    "TAB", "TABS", "TABLET", "TABLETS",
    "CAP", "CAPS", "CAPSULE", "CAPSULES",
    "COMP", "COMPRIME", "COMPRIMES", "COMPRIMÉ", "COMPRIMÉS",
    "SUSP", "SUSPENSION", "SOLUTION", "SYRUP", "SIROP",
    "CREME", "CRÈME", "POMMADE", "GEL", "LOTION",
    "INJ", "INJECTION", "VIAL", "VIALS", "AMPOULE", "PATCH", "SPRAY",
    "DROPS", "GOUTTES", "BOTTLE", "BOTTLES", "BOX", "BOXES", "PACK", "PACKS",
    "PLAQUETTE", "BANDELETTE", "STYLO", "PEN", "PENS",

    # release / dosage markers
    "XR", "SR", "ER", "CR", "DR", "ODT",

    # measurement units
    "MG", "MCG", "G", "KG", "ML", "L", "IU", "U", "%",

    # annoying non-molecule starts
    "ACIDE", "ROUGE", "MENTHE",
}

UNIT_WORDS = {
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
    "PEN": "pens",
    "PENS": "pens",
    "STYLO": "pens",

    "ML": "mL",
    "L": "L",
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


def is_mostly_numeric(series: pd.Series, sample_n: int = 300) -> float:
    vals = series.dropna().head(sample_n)
    if len(vals) == 0:
        return 0.0
    hits = sum(1 for v in vals if to_float_or_none(v) is not None)
    return hits / len(vals)


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


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), "Loaded CSV"

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file), "Loaded Excel"

    if name.endswith(".pdf"):
        return read_pdf_to_df(uploaded_file), "Loaded PDF (via pdfplumber)"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, or PDF.")


# -----------------------------
# Column helpers
# -----------------------------
def guess_quantity_columns(df: pd.DataFrame) -> List[Tuple[float, int]]:
    scored = []
    for i in range(df.shape[1]):
        score = is_mostly_numeric(df.iloc[:, i])
        scored.append((score, i))
    scored.sort(reverse=True)
    return scored


def guess_text_columns(df: pd.DataFrame, qty_idx: int) -> List[int]:
    candidates = []
    for i in range(df.shape[1]):
        if i == qty_idx:
            continue
        sample = df.iloc[:, i].dropna().astype(str).head(400)
        if len(sample) == 0:
            continue
        avg_len = sum(len(v.strip()) for v in sample) / len(sample)
        non_num_frac = sum(1 for v in sample if to_float_or_none(v) is None) / len(sample)
        score = avg_len * non_num_frac
        candidates.append((score, i))
    candidates.sort(reverse=True)
    return [i for _, i in candidates[:6]]


def build_description(row: pd.Series, text_cols: List[int]) -> str:
    parts = []
    for i in text_cols:
        val = row.iloc[i]
        if val is None:
            continue
        s = norm(val)
        if not s:
            continue
        parts.append(s)
    return norm(" ".join(parts))


# -----------------------------
# Cleaning logic
# -----------------------------
def remove_vendors_and_manufacturer(text: str, vendors: List[str]) -> Tuple[str, Set[str]]:
    """
    Example:
    'APOTEX McKesson APO FLUOXETINE 20 MG CAPS'
    -> remove everything before vendor + vendor itself
    -> 'APO FLUOXETINE 20 MG CAPS'
    """
    s = norm(text)
    found_vendors = set()

    if not s:
        return s, found_vendors

    earliest_match = None
    chosen_vendor = None

    for v in sorted({norm(x) for x in vendors if norm(x)}, key=len, reverse=True):
        m = re.search(rf"(?i)\b{re.escape(v)}\b", s)
        if m:
            found_vendors.add(v.upper())
            if earliest_match is None or m.start() < earliest_match:
                earliest_match = m.start()
                chosen_vendor = v

    if earliest_match is not None and chosen_vendor is not None:
        s = norm(s[earliest_match + len(chosen_vendor):])

    for v in sorted({norm(x) for x in vendors if norm(x)}, key=len, reverse=True):
        if re.search(rf"(?i)\b{re.escape(v)}\b", s):
            found_vendors.add(v.upper())
            s = re.sub(rf"(?i)\b{re.escape(v)}\b", " ", s)

    return norm(s), found_vendors


def strip_generic_prefix(text: str, prefixes: Set[str]) -> Tuple[Optional[str], str]:
    """
    Only strips prefix at the START.
    Examples:
    APO-FLUOXETINE -> FLUOXETINE
    APO FLUOXETINE -> FLUOXETINE
    """
    s = norm(text)
    if not s:
        return None, s

    m = re.match(r"^([A-Za-z]{1,15})\s*-\s*(.+)$", s)
    if m:
        pfx = m.group(1).upper()
        rest = norm(m.group(2))
        if pfx in prefixes:
            return pfx, rest

    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in prefixes:
            return first, norm(" ".join(parts[1:]))

    return None, s


def detect_probable_unit(text: str) -> str:
    s = norm(text).upper()
    for raw, label in UNIT_WORDS.items():
        if re.search(rf"\b{re.escape(raw)}\b", s):
            return label
    return "units"


def extract_molecule(text: str) -> str:
    """
    Pull first real molecule-like token from the cleaned description.
    We avoid short junk, units, vendor/company terms, and numeric tokens.
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
def aggregate_inventory(
    df: pd.DataFrame,
    qty_idx: int,
    text_cols: List[int],
    vendors: List[str],
    prefixes: Set[str],
):
    totals: Dict[str, float] = defaultdict(float)
    unit_labels: Dict[str, List[str]] = defaultdict(list)
    sample_desc: Dict[str, str] = {}
    prefix_seen: Dict[str, Set[str]] = defaultdict(set)
    vendor_seen: Dict[str, Set[str]] = defaultdict(set)

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

        cleaned_desc, found_vendors = remove_vendors_and_manufacturer(full_desc, vendors)
        found_prefix, desc_no_prefix = strip_generic_prefix(cleaned_desc, prefixes)
        molecule = extract_molecule(desc_no_prefix)

        if not molecule:
            skipped += 1
            continue

        probable_unit = detect_probable_unit(desc_no_prefix)

        totals[molecule] += float(qty)
        unit_labels[molecule].append(probable_unit)

        if molecule not in sample_desc:
            sample_desc[molecule] = desc_no_prefix
        if found_prefix:
            prefix_seen[molecule].add(found_prefix)
        if found_vendors:
            vendor_seen[molecule].update(found_vendors)

        parsed += 1

    rows = []
    for molecule in sorted(totals.keys()):
        unit_counter = Counter(unit_labels[molecule])
        most_common_unit = unit_counter.most_common(1)[0][0] if unit_counter else "units"

        rows.append(
            {
                "Molécule": molecule,
                "Quantité totale en stock": totals[molecule],
                "Unité probable": most_common_unit,
                "Exemple description": sample_desc.get(molecule, ""),
                "Préfixes trouvés": ", ".join(sorted(prefix_seen.get(molecule, set()))),
                "Vendeurs trouvés": ", ".join(sorted(vendor_seen.get(molecule, set()))),
            }
        )

    out = pd.DataFrame(rows)
    return out, parsed, skipped


# -----------------------------
# Optional PDF export
# -----------------------------
def dataframe_to_pdf(df: pd.DataFrame, title: str) -> bytes:
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

    c.drawString(x, y, f"{'Molécule':20} {'Quantité':>12}  {'Unité':14}")
    y -= line_h
    c.drawString(x, y, "-" * 60)
    y -= line_h

    for _, row in df.iterrows():
        line = f"{str(row['Molécule'])[:20]:20} {float(row['Quantité totale en stock']):>12.2f}  {str(row['Unité probable'])[:14]}"
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
st.write("Choisis la vraie colonne de quantité en stock. That is the whole game.")

uploaded = st.file_uploader(
    "Upload inventory file (CSV / Excel / PDF)",
    type=["csv", "xlsx", "xls", "pdf"]
)

with st.sidebar:
    st.header("Vendeurs à retirer du nom")
    vendors_text = st.text_area(
        "Une entrée par ligne",
        value="\n".join(DEFAULT_VENDORS),
        height=120
    )
    vendors = [norm(v) for v in vendors_text.splitlines() if norm(v)]

    st.divider()
    st.header("Préfixes génériques à ignorer")
    prefixes_text = st.text_area(
        "Une entrée par ligne",
        value="\n".join(DEFAULT_PREFIXES),
        height=220
    )
    prefixes = {norm(p).upper() for p in prefixes_text.splitlines() if norm(p)}

    st.divider()
    show_debug = st.checkbox("Afficher colonnes debug", value=False)
    export_pdf = st.checkbox("Générer aussi un PDF", value=False)

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

# Manual quantity selection
st.subheader("1) Choisis la colonne de quantité")
qty_guesses = guess_quantity_columns(df_raw)
guess_text = ", ".join([f"Col {i} ({score:.2f})" for score, i in qty_guesses[:6]])
st.caption(f"Possible numeric columns: {guess_text}")

qty_idx = st.selectbox(
    "Select the column that contains stock quantity",
    options=list(range(df_raw.shape[1])),
    index=qty_guesses[0][1] if qty_guesses else 0,
    format_func=lambda x: f"Column {x}"
)

# Text columns selection
st.subheader("2) Colonnes de description")
suggested_text_cols = guess_text_columns(df_raw, qty_idx)

text_cols = st.multiselect(
    "Select the columns that together describe the medication",
    options=list(range(df_raw.shape[1])),
    default=suggested_text_cols,
    format_func=lambda x: f"Column {x}"
)

if not text_cols:
    st.warning("Pick at least one description column.")
    st.stop()

out, parsed, skipped = aggregate_inventory(
    df_raw,
    qty_idx=qty_idx,
    text_cols=text_cols,
    vendors=vendors,
    prefixes=prefixes,
)

st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

display_df = out.copy()
if not show_debug and not display_df.empty:
    display_df = display_df[["Molécule", "Quantité totale en stock", "Unité probable"]]

st.subheader("Résultat")
st.dataframe(display_df, use_container_width=True)

# Excel download
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Inventaire_par_molecule")

st.download_button(
    "Télécharger Excel (.xlsx)",
    data=excel_buf.getvalue(),
    file_name="inventaire_par_molecule.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Optional PDF download
if export_pdf:
    try:
        pdf_bytes = dataframe_to_pdf(display_df, "Inventaire par molécule")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_par_molecule.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
