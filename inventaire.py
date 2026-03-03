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
# Prefix seed (Canada-typical)
# Sources/examples:
# - Ontario ODB interchangeable listings show Apo-, Auro-, Jamp-, Mint-, Mylan-, etc. :contentReference[oaicite:2]{index=2}
# - CIHI notes examples NOVO, APO, PMS, RATIO, SANDOZ :contentReference[oaicite:3]{index=3}
# This is NOT “everything in Canada”, so we ALSO learn prefixes from your file.
# -----------------------------
CANADA_PREFIX_SEED = {
    "APO", "PMS", "RATIO", "SANDOZ", "NOVO",
    "AURO", "JAMP", "MINT", "MYLAN",
    "ACH", "ACC", "AG", "MAR", "NRA", "CO", "PRO", "TEVA", "TARO", "ACT",
}

# Vendors: removed from name, stored separately.
DEFAULT_VENDORS = {
    "MCKESSON",
    "PHARMA PLUS",
    "PHARMAPLUS",
    "PHARMA+",
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
# Vendor extraction (remove from name, keep separate)
# -----------------------------
def extract_and_remove_vendors(text: str, vendors: Set[str]) -> Tuple[str, Set[str]]:
    s = normalise_space(text)
    found = set()
    if not s:
        return s, found

    # remove longer phrases first
    for v in sorted({normalise_space(x).upper() for x in vendors if normalise_space(x)}, key=len, reverse=True):
        pattern = rf"(?i)\b{re.escape(v)}\b"
        if re.search(pattern, s):
            found.add(v)
            s = re.sub(pattern, " ", s)

    return normalise_space(s), found


# -----------------------------
# Prefix learning + stripping for grouping
# -----------------------------
def learn_hyphen_prefixes(desc_series: pd.Series) -> Set[str]:
    prefixes = set()
    for raw in desc_series.dropna().astype(str).head(20000):
        s = normalise_space(raw)
        m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
        if m:
            p = m.group(1).upper()
            if p.isalpha():
                prefixes.add(p)
    return prefixes


def strip_prefix(desc: str, prefix_set: Set[str]) -> Tuple[Optional[str], str]:
    """
    Returns (prefix_found, core_name)
    Strips:
      - Hyphen prefixes always (AA-FLUOXETINE)
      - Space prefixes if in prefix_set (PMS FLUOXETINE)
    """
    s = normalise_space(desc)
    if not s:
        return None, s

    # Hyphen
    m = re.match(r"^([A-Za-z]{1,10})\s*-\s*(.+)$", s)
    if m:
        p = m.group(1).upper()
        rest = normalise_space(m.group(2))
        if p.isalpha() and rest:
            return p, rest

    # Space
    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in prefix_set:
            return first, normalise_space(" ".join(parts[1:]))

    return None, s


def extract_molecule(core: str) -> str:
    """
    Groups by the molecule name (your request: venlafaxine, pregabalin, etc.)
    Take the first “meaningful” token that contains letters.
    Keeps combos like AMOXICILLIN/CLAVULANATE or A+B.
    """
    s = normalise_space(core)
    if not s:
        return s

    # Split and find first token with letters
    for tok in s.split():
        if re.search(r"[A-Za-zÀ-ÿ]", tok):
            # clean punctuation edges
            tok = tok.strip(" ,;()[]{}")
            return tok.upper()
    return s.upper()


def is_reasonable_core(core: str) -> bool:
    c = normalise_space(core)
    return bool(re.search(r"[A-Za-zÀ-ÿ]", c)) and len(c) >= 4


# -----------------------------
# Aggregation
# -----------------------------
def aggregate(df: pd.DataFrame, qty_idx: int, vendors: Set[str]) -> Tuple[pd.DataFrame, int, int, Set[str], Set[str]]:
    text_cols = detect_text_columns(df, qty_idx)

    # Build descriptions (cleaned for vendor words) to learn prefixes safely
    tmp_desc = []
    for _, row in df.iterrows():
        d = build_description(row, text_cols)
        d, _ = extract_and_remove_vendors(d, vendors)
        if d:
            tmp_desc.append(d)

    hyphen_prefixes = learn_hyphen_prefixes(pd.Series(tmp_desc))
    prefix_set = set(CANADA_PREFIX_SEED) | hyphen_prefixes  # seed + learned

    # Aggregate by molecule
    totals: Dict[str, float] = {}
    brand_prefixes: Dict[str, Set[str]] = {}
    vendor_seen: Dict[str, Set[str]] = {}

    parsed = 0
    skipped = 0

    for _, row in df.iterrows():
        qty = to_float_or_none(row.iloc[qty_idx])
        if qty is None:
            skipped += 1
            continue

        desc = build_description(row, text_cols)
        desc, vend = extract_and_remove_vendors(desc, vendors)

        if not desc:
            skipped += 1
            continue

        pfx, core = strip_prefix(desc, prefix_set)

        if not is_reasonable_core(core):
            skipped += 1
            continue

        molecule = extract_molecule(core)
        if not molecule:
            skipped += 1
            continue

        totals[molecule] = totals.get(molecule, 0.0) + float(qty)

        if pfx:
            brand_prefixes.setdefault(molecule, set()).add(pfx.upper())
        if vend:
            vendor_seen.setdefault(molecule, set()).update({v.upper() for v in vend})

        parsed += 1

    out = pd.DataFrame(
        {
            "Molécule (regroupée)": list(totals.keys()),
            "Quantité totale": [totals[k] for k in totals.keys()],
            "Préfixes / Marques trouvés": [", ".join(sorted(brand_prefixes.get(k, set()))) for k in totals.keys()],
            "Vendeurs trouvés": [", ".join(sorted(vendor_seen.get(k, set()))) for k in totals.keys()],
        }
    ).sort_values("Molécule (regroupée)").reset_index(drop=True)

    out["Préfixes / Marques trouvés"] = out["Préfixes / Marques trouvés"].fillna("")
    out["Vendeurs trouvés"] = out["Vendeurs trouvés"].fillna("")

    return out, parsed, skipped, prefix_set, hyphen_prefixes


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

    cols = ["Molécule (regroupée)", "Quantité totale", "Préfixes / Marques trouvés", "Vendeurs trouvés"]
    c.drawString(x, y, f"{cols[0][:20]:20} {cols[1]:>12}  {cols[2][:25]:25}  {cols[3][:25]}")
    y -= lh
    c.drawString(x, y, "-" * 115)
    y -= lh

    for _, r in df.iterrows():
        mol = str(r[cols[0]])[:20]
        qty = f"{float(r[cols[1]]):.2f}"
        pfx = str(r[cols[2]])[:25]
        ven = str(r[cols[3]])[:25]
        line = f"{mol:20} {qty:>12}  {pfx:25}  {ven}"

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
st.title("Inventaire — Regroupement des génériques (préfixes fusionnés) + vendeurs séparés")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Vendeurs (seront retirés du nom et mis dans une colonne)")
    vendors_text = st.text_area(
        "Une entrée par ligne",
        value="\n".join(sorted(DEFAULT_VENDORS)),
        height=120,
    )
    vendors = {normalise_space(v).upper() for v in vendors_text.splitlines() if normalise_space(v)}

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

# Quantity detection with fallback
candidates = detect_quantity_candidates(df_raw)
best_score, best_idx = candidates[0]
st.caption(f"Auto quantity guess: column {best_idx} (numeric score {best_score:.2f})")

qty_idx = best_idx
if best_score < 0.30:
    st.warning("Quantity auto-detection is weak. Pick the correct quantity column.")
    options = [f"Col {i} (score {s:.2f})" for s, i in candidates[: min(10, len(candidates))]]
    choice = st.selectbox("Select quantity column", options, index=0)
    qty_idx = int(re.search(r"Col (\d+)", choice).group(1))

out, parsed, skipped, prefix_set, hyphen_prefixes = aggregate(df_raw, qty_idx, vendors)

st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

with st.expander("Debug: prefixes utilisés"):
    st.write("Seed+learned prefix set size:", len(prefix_set))
    st.write(", ".join(sorted(prefix_set))[:4000])

with st.expander("Debug: prefixes appris par forme AA-XXX"):
    st.write(", ".join(sorted(hyphen_prefixes))[:4000])

st.subheader("Résultat (regroupé par molécule)")
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
        pdf_bytes = dataframe_to_simple_pdf(out, "Inventaire regroupé (molécule)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_regroupe.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
