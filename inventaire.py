# inventaire.py
# pip install streamlit pandas openpyxl pdfplumber pdfminer.six
# Optional PDF export: pip install reportlab

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
# Seed prefixes (Canada-typical, not “APO-only”)
# This is a STARTER set; the app also learns from YOUR file automatically.
# -----------------------------
CANADA_PREFIX_SEED = {
    "AA", "APO", "PMS", "RATIO", "TEVA", "SANDOZ", "AURO", "JAMP", "MINT",
    "MYLAN", "TARO", "ACT", "ACC", "NOVO", "RAN", "PHL", "BIO", "OPUS",
}


# -----------------------------
# Utils
# -----------------------------
def normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def looks_numeric_cell(x) -> bool:
    s = str(x).strip()
    if not s:
        return False
    s = s.replace(",", ".")
    return bool(re.search(r"-?\d+(?:\.\d+)?", s))


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
            for table in page.extract_tables() or []:
                for row in table or []:
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
        col = df.iloc[:, i].dropna().head(400)
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
        sample = df.iloc[:, i].dropna().astype(str).head(250)
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
# Prefix learning + stripping
# -----------------------------
def guess_prefixes_from_descriptions(desc_series: pd.Series, min_count: int = 3) -> Set[str]:
    """
    Learns candidate prefixes from your file.
    Rules:
      - take first token OR token before '-' if present
      - alphabetic only
      - short-ish (1–8 chars)
      - appears >= min_count
    """
    counts: Dict[str, int] = {}

    for raw in desc_series.dropna().astype(str).head(5000):
        s = normalise_space(raw)
        if not s:
            continue

        # token before '-' is strong signal (e.g., "PMS-FLUOXETINE")
        m = re.match(r"^([A-Za-z]{1,8})\s*-\s*(.+)$", s)
        if m:
            p = m.group(1).upper()
        else:
            p = s.split()[0].upper()

        if p.isalpha() and 1 <= len(p) <= 8:
            counts[p] = counts.get(p, 0) + 1

    return {p for p, c in counts.items() if c >= min_count}


def split_prefix_and_core(desc: str, known_prefixes: Set[str]) -> Tuple[Optional[str], str]:
    """
    Accepts:
      - "PMS-FLUOXETINE ..."  -> ("PMS", "FLUOXETINE ...")
      - "PMS FLUOXETINE ..."  -> ("PMS", "FLUOXETINE ...")
      - "aa-fluoxetine"       -> ("AA", "FLUOXETINE")
    Only strips if prefix is in known_prefixes.
    """
    s = normalise_space(desc)
    if not s:
        return None, s

    # hyphen form
    m = re.match(r"^([A-Za-z]{1,8})\s*-\s*(.+)$", s)
    if m:
        p = m.group(1).upper()
        rest = m.group(2)
        if p in known_prefixes:
            return p, normalise_space(rest)

    # space form
    first = s.split()[0].upper()
    if first in known_prefixes:
        rest = " ".join(s.split()[1:])
        return first, normalise_space(rest)

    return None, s


# -----------------------------
# Aggregate with brand/prefix list
# -----------------------------
def aggregate_inventory_with_brands(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int, Set[str]]:
    qty_idx = detect_quantity_column(df)
    text_cols = detect_text_columns(df, qty_idx)

    # Build quick descriptions to learn prefixes from THIS file
    tmp_desc = []
    for _, row in df.iterrows():
        d = build_description(row, text_cols)
        if d:
            tmp_desc.append(d)
    desc_series = pd.Series(tmp_desc)

    learned = guess_prefixes_from_descriptions(desc_series, min_count=3)
    known_prefixes = set(CANADA_PREFIX_SEED) | learned

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

        prefix, core = split_prefix_and_core(desc, known_prefixes)

        # Filter out garbage rows like only units/packaging fragments
        if not re.search(r"[A-Za-zÀ-ÿ]", core) or len(core) < 6:
            skipped += 1
            continue

        totals[core] = totals.get(core, 0.0) + float(qty)

        if prefix:
            brands.setdefault(core, set()).add(prefix)

        parsed += 1

    out = pd.DataFrame(
        {
            "Médicament (nom regroupé)": list(totals.keys()),
            "Quantité totale": [totals[k] for k in totals.keys()],
            "Marques / Préfixes trouvés": [", ".join(sorted(brands.get(k, set()))) for k in totals.keys()],
        }
    ).sort_values("Médicament (nom regroupé)").reset_index(drop=True)

    return out, qty_idx, parsed, skipped, known_prefixes


# -----------------------------
# PDF export (optional)
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

    headers = ["Médicament (nom regroupé)", "Quantité totale", "Marques / Préfixes trouvés"]
    c.drawString(x, y, f"{headers[0][:55]:55}  {headers[1]:>12}  {headers[2]}")
    y -= lh
    c.drawString(x, y, "-" * 110)
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
st.title("Inventaire — Regroupement des génériques (préfixes → même médicament)")

uploaded = st.file_uploader("Upload inventory file (CSV / Excel / PDF)", type=["csv", "xlsx", "xls", "pdf"])

with st.sidebar:
    st.header("Contrôles (optionnels)")
    extra_add = st.text_input("Ajouter des préfixes (comma-separated)", value="")
    extra_remove = st.text_input("Forcer à NE PAS traiter comme préfixe (comma-separated)", value="")

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

out, qty_idx, parsed, skipped, known_prefixes = aggregate_inventory_with_brands(df_raw)

# Apply optional user tweaks AFTER detection
add_set = {p.strip().upper() for p in extra_add.split(",") if p.strip()}
remove_set = {p.strip().upper() for p in extra_remove.split(",") if p.strip()}

if add_set or remove_set:
    # Re-run aggregation with modified known_prefixes
    # (quick re-run: we just hack known_prefixes and recompute)
    # For simplicity: rebuild with the adjusted set by calling a modified version inline.
    # ---- minimal recompute ----
    qty_idx = detect_quantity_column(df_raw)
    text_cols = detect_text_columns(df_raw, qty_idx)
    adjusted = (known_prefixes | add_set) - remove_set

    totals, brands = {}, {}
    parsed, skipped = 0, 0
    for _, row in df_raw.iterrows():
        qty = to_float_or_none(row.iloc[qty_idx])
        if qty is None:
            skipped += 1
            continue
        desc = build_description(row, text_cols)
        if not desc:
            skipped += 1
            continue
        prefix, core = split_prefix_and_core(desc, adjusted)
        if not re.search(r"[A-Za-zÀ-ÿ]", core) or len(core) < 6:
            skipped += 1
            continue
        totals[core] = totals.get(core, 0.0) + float(qty)
        if prefix:
            brands.setdefault(core, set()).add(prefix)
        parsed += 1

    out = pd.DataFrame(
        {
            "Médicament (nom regroupé)": list(totals.keys()),
            "Quantité totale": [totals[k] for k in totals.keys()],
            "Marques / Préfixes trouvés": [", ".join(sorted(brands.get(k, set()))) for k in totals.keys()],
        }
    ).sort_values("Médicament (nom regroupé)").reset_index(drop=True)

st.caption(f"Auto-detected quantity column: {qty_idx}")
st.write(f"Parsed rows: **{parsed}** — Skipped rows: **{skipped}**")

with st.expander("Préfixes détectés / utilisés (debug)"):
    st.write(", ".join(sorted(known_prefixes))[:2000])

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
        pdf_bytes = dataframe_to_simple_pdf(out, "Inventaire regroupé (préfixes → même médicament)")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_regroupe.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
