# inventaire.py
# Upload inventory -> auto-group by medication -> ignore prefixes -> total stock units
# Keeps dosage/strength in the medication name.
#
# Install:
#   pip install streamlit pandas openpyxl pdfplumber pdfminer.six
#
# Optional PDF export:
#   pip install reportlab
#
# Run:
#   streamlit run inventaire.py

import io
import re
from collections import defaultdict
from typing import Optional, Tuple, List

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
# Prefixes to ignore at the start
# -----------------------------
IGNORED_PREFIXES = {
    "AA", "APO", "PMS", "RATIO", "SANDOZ", "TEVA", "AURO", "JAMP", "MINT",
    "MYLAN", "TARO", "ACT", "NOVO", "BIO", "OPUS", "RAN", "MAR",
    "ACC", "ACH", "AG", "BGP"
}

# Tokens that mean packaging / form and should usually stop the name
STOP_TOKENS = {
    "TAB", "TABS", "TABLET", "TABLETS",
    "CAP", "CAPS", "CAPSULE", "CAPSULES",
    "COMP", "COMPRIME", "COMPRIMES", "COMPRIMÉ", "COMPRIMÉS",
    "DOSE", "DOSES",
    "INH", "INHALATEUR", "SPRAY", "POUDRE", "SUSP", "SUSPENSION",
    "SOLUTION", "SYRUP", "SIROP", "GEL", "CREME", "CRÈME",
    "XR", "SR", "ER", "CR", "DR", "CD",
    "PLAQUETTE", "ENTERIC", "ENT", "ENT.",
    "POMPE", "PUMP", "BOUTEILLE", "BOTTLE", "BOTTLES",
    "VIAL", "VIALS", "AMP", "AMPOULE", "PATCH", "PATCHES",
    "STYLO", "PEN", "PENS"
}

UNIT_WORDS = {
    "TAB": "pills",
    "TABS": "pills",
    "TABLET": "pills",
    "TABLETS": "pills",
    "COMP": "pills",
    "COMPRIME": "pills",
    "COMPRIMES": "pills",
    "COMPRIMÉ": "pills",
    "COMPRIMÉS": "pills",

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
    "DOSE": "doses",
    "DOSES": "doses",
}

# Single-word starts that should keep the next token too
KEEP_SECOND_TOKEN = {
    "VITAMINE", "ACIDE"
}


# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def to_float_or_zero(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return 0.0
    try:
        return float(m.group(0))
    except Exception:
        return 0.0


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm(str(c)).replace("\n", " ") for c in df.columns]
    return df


# -----------------------------
# Read files
# -----------------------------
def read_csv_inventory(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, skiprows=1)
    return clean_columns(df)


def read_excel_inventory(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df = clean_columns(df)

    if "Produit" not in df.columns and "Qté servie" not in df.columns:
        df = pd.read_excel(uploaded_file, skiprows=1)
        df = clean_columns(df)

    return df


def read_pdf_inventory(uploaded_file) -> pd.DataFrame:
    if not PDFPLUMBER_OK:
        raise RuntimeError("PDF support not installed. Add pdfplumber to requirements.txt.")

    rows = []
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
        raise RuntimeError("No tables detected in PDF.")

    rows = [r + [None] * (max_cols - len(r)) for r in rows]
    raw = pd.DataFrame(rows)

    header_row_idx = None
    for i in range(min(10, len(raw))):
        joined = " | ".join([norm(x).replace("\n", " ") for x in raw.iloc[i].fillna("").tolist()])
        if "Produit" in joined and "Qté servie" in joined:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError("Could not detect inventory headers in PDF.")

    header = [norm(x).replace("\n", " ") for x in raw.iloc[header_row_idx].tolist()]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = header
    return clean_columns(df)


def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return read_csv_inventory(uploaded_file), "Loaded CSV"
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return read_excel_inventory(uploaded_file), "Loaded Excel"
    if name.endswith(".pdf"):
        return read_pdf_inventory(uploaded_file), "Loaded PDF"

    raise RuntimeError("Unsupported file type. Upload CSV, XLSX, XLS, or PDF.")


# -----------------------------
# Medication parsing
# -----------------------------
def strip_prefix(product_name: str) -> Tuple[Optional[str], str]:
    """
    Examples:
      APO FLUOXETINE 20MG COMPRIME -> FLUOXETINE 20MG COMPRIME
      APO-FLUOXETINE 20MG -> FLUOXETINE 20MG
      AA LEVOCARB CR 200+50MG -> LEVOCARB CR 200+50MG
    """
    s = norm(product_name).upper().replace("\n", " ")
    if not s:
        return None, s

    m = re.match(r"^([A-Z]{1,15})\s*-\s*(.+)$", s)
    if m:
        prefix = m.group(1).upper()
        rest = norm(m.group(2))
        if prefix in IGNORED_PREFIXES:
            return prefix, rest

    parts = s.split()
    if len(parts) >= 2:
        first = parts[0].upper()
        if first in IGNORED_PREFIXES:
            return first, norm(" ".join(parts[1:]))

    return None, s


def is_strength_token(tok: str) -> bool:
    """
    Accept dosage/strength tokens in the medication name.
    Examples:
      20MG
      1000
      IU
      0.5%
      200+50MG
      5/325MG
      10MCG
    """
    tok = tok.upper().strip()

    if not tok:
        return False

    # plain numbers that might belong to dosage
    if re.fullmatch(r"\d+(?:[.,]\d+)?", tok):
        return True

    # IU / % / MG / MCG / G / ML / L forms
    if re.fullmatch(r"\d+(?:[.,]\d+)?(?:MG|MCG|G|KG|ML|L|IU|U|%)", tok):
        return True

    # combo strengths
    if re.fullmatch(r"\d+(?:[.,]\d+)?[+/]\d+(?:[.,]\d+)?(?:MG|MCG|G|ML|IU|%)?", tok):
        return True

    # fraction strengths
    if re.fullmatch(r"\d+(?:[.,]\d+)?/\d+(?:[.,]\d+)?(?:MG|MCG|G|ML|IU|%)?", tok):
        return True

    # standalone unit after number token
    if tok in {"MG", "MCG", "G", "KG", "ML", "L", "IU", "U", "%"}:
        return True

    return False


def extract_medication_key(product_name: str) -> str:
    """
    Keep the medication name + useful strength/dosage.
    Examples:
      APO FLUOXETINE 20MG CAP -> FLUOXETINE 20MG
      VITAMINE D 1000 IU TAB -> VITAMINE D 1000 IU
      M LATANOPROST-TIMOLOL 0.005%-0.5% -> LATANOPROST-TIMOLOL 0.005%-0.5%
    """
    _, cleaned = strip_prefix(product_name)
    cleaned = norm(cleaned).upper()

    words = cleaned.split()
    tokens = []
    saw_name = False

    i = 0
    while i < len(words):
        raw_tok = words[i]
        tok = raw_tok.strip(" ,;()[]{}")

        if not tok:
            i += 1
            continue

        # ignore lonely junk tokens before the real name
        if not saw_name and len(tok.strip("+-/")) <= 1:
            i += 1
            continue

        # stop at package/form tokens
        if tok.upper() in STOP_TOKENS:
            break

        # if we haven't started yet, we need a real word
        if not saw_name:
            if re.search(r"[A-Z]", tok):
                tokens.append(tok.upper())
                saw_name = True

                # special cases like VITAMINE D / ACIDE FOLIQUE
                if tok.upper() in KEEP_SECOND_TOKEN and i + 1 < len(words):
                    nxt = words[i + 1].strip(" ,;()[]{}").upper()
                    if nxt and nxt not in STOP_TOKENS and len(nxt) > 1:
                        tokens.append(nxt)
                        i += 1
            i += 1
            continue

        # after name started, keep strength tokens too
        if is_strength_token(tok):
            tokens.append(tok.upper())
            i += 1
            continue

        # if it's another normal word after the first name, keep it only if
        # it still looks like part of the med name and not form/packaging junk
        if re.search(r"[A-Z]", tok) and tok.upper() not in STOP_TOKENS:
            # avoid meaningless one-letter junk
            if len(tok.strip("+-/")) > 1:
                # allow chemical-family style continuations
                if len(tokens) < 3 and not any(is_strength_token(t) for t in tokens):
                    tokens.append(tok.upper())
                    i += 1
                    continue

        break

    # fallback if parsing was too aggressive
    if not tokens:
        for raw_tok in words:
            tok = raw_tok.strip(" ,;()[]{}").upper()
            if tok and len(tok) > 1 and re.search(r"[A-Z]", tok):
                return tok
        return ""

    return " ".join(tokens)


def detect_probable_unit(product_name: str, format_value: str) -> str:
    text = f"{norm(product_name)} {norm(format_value)}".upper()

    if re.search(r"\b(DOSE|DOSES)\b", text):
        return "doses"
    if re.search(r"\b(CAP|CAPS|CAPSULE|CAPSULES)\b", text):
        return "capsules"
    if re.search(r"\b(TAB|TABS|TABLET|TABLETS|COMP|COMPRIME|COMPRIMES|COMPRIMÉ|COMPRIMÉS)\b", text):
        return "pills"
    if re.search(r"\b(ML|L)\b", text):
        return "mL"
    return "units"


# -----------------------------
# Inventory logic
# -----------------------------
def normalise_inventory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DIN" in df.columns:
        df = df[df["DIN"].astype(str).str.upper() != "DIN"]

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    needed = ["Produit", "Format", "Nombre de services", "Qté servie"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return df


def aggregate_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_inventory_dataframe(df)

    totals = defaultdict(float)
    prefixes = defaultdict(set)
    units = defaultdict(list)
    examples = {}

    for _, row in df.iterrows():
        product = norm(row["Produit"])
        fmt = norm(row.get("Format", ""))

        if not product:
            continue

        prefix, _ = strip_prefix(product)
        med_key = extract_medication_key(product)

        if not med_key or len(med_key) <= 1:
            continue

        n_services = to_float_or_zero(row.get("Nombre de services", 0))
        qty_served = to_float_or_zero(row.get("Qté servie", 0))
        total_units = n_services * qty_served

        if total_units <= 0:
            continue

        totals[med_key] += total_units

        if prefix:
            prefixes[med_key].add(prefix)

        units[med_key].append(detect_probable_unit(product, fmt))

        if med_key not in examples:
            examples[med_key] = product

    rows = []
    for med in sorted(totals.keys()):
        unit = max(set(units[med]), key=units[med].count) if units[med] else "units"
        rows.append({
            "Médicament": med,
            "Quantité totale": totals[med],
            "Unité probable": unit,
            "Exemple": examples.get(med, ""),
            "Préfixes ignorés trouvés": ", ".join(sorted(prefixes.get(med, set()))),
        })

    return pd.DataFrame(rows)


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
    line_h = 10

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 8)
    c.drawString(x, y, f"{'Médicament':34} {'Quantité':>12}  {'Unité':12}")
    y -= line_h
    c.drawString(x, y, "-" * 78)
    y -= line_h

    for _, row in df.iterrows():
        line = f"{str(row['Médicament'])[:34]:34} {float(row['Quantité totale']):>12.2f}  {str(row['Unité probable'])[:12]}"
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
st.set_page_config(page_title="Inventaire regroupé", layout="wide")
st.title("Inventaire regroupé")
st.write("Upload le fichier. L’app regroupe automatiquement les médicaments en ignorant les prefixes comme APO / PMS / AURO / JAMP / AA, tout en gardant le dosage utile dans le nom.")

uploaded = st.file_uploader(
    "Upload inventory file (CSV / Excel / PDF)",
    type=["csv", "xlsx", "xls", "pdf"]
)

with st.sidebar:
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

try:
    result = aggregate_inventory(df_raw)
except Exception as e:
    st.error(f"Failed to process inventory: {e}")
    st.stop()

if result.empty:
    st.warning("No medication quantities could be calculated from this file.")
    st.stop()

display_df = result.copy()
if not show_debug:
    display_df = display_df[["Médicament", "Quantité totale", "Unité probable"]]

st.subheader("Résultat")
st.dataframe(display_df, use_container_width=True)

# Excel output
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    result.to_excel(writer, index=False, sheet_name="Inventaire_regroupe")

st.download_button(
    "Télécharger Excel (.xlsx)",
    data=excel_buf.getvalue(),
    file_name="inventaire_regroupe.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Optional PDF output
if export_pdf:
    try:
        pdf_bytes = dataframe_to_pdf(display_df, "Inventaire regroupé")
        st.download_button(
            "Télécharger PDF (.pdf)",
            data=pdf_bytes,
            file_name="inventaire_regroupe.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(str(e))
