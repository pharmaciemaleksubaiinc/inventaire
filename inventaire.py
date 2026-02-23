# app.py
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Generic Inventory Totals", layout="wide")

PREFIXES = {
    "APO", "SANDOZ", "TEVA", "MYLAN", "AURO", "PMS", "ACT", "RIVA", "RANBAXY",
    "JAMP", "PHARMASCIENCE", "MINT", "NOVA", "SANIS", "RATIO", "GENPHARM",
    "DOM", "MAR", "AA", "BIO", "ACCEL", "TARO"
}

PREFIX_RE = re.compile(r"^([A-Z0-9]+)[\-/]?$")

def normalise_med_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""

    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if not parts:
        return s

    first = parts[0].upper()
    m = PREFIX_RE.match(first)
    first_clean = m.group(1) if m else first

    if first_clean in PREFIXES:
        return " ".join(parts[1:]).strip()
    return s

def detect_columns(df: pd.DataFrame):
    if df.empty or df.shape[1] < 2:
        raise ValueError("CSV is empty or has too few columns.")

    # Best qty col = highest fraction numeric
    best_qty_col, best_score = None, -1
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        score = numeric.notna().mean()
        if score > best_score:
            best_score, best_qty_col = score, col

    # Best name col = text col with longest average string length
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        best_name_col = max(
            text_cols,
            key=lambda c: df[c].astype(str).str.len().replace("nan", "").mean()
        )
    else:
        best_name_col = df.columns[0]

    if best_name_col == best_qty_col:
        best_name_col = df.columns[0] if df.columns[0] != best_qty_col else df.columns[1]

    return best_name_col, best_qty_col

def summarise(df: pd.DataFrame, name_col: str, qty_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_name"] = work[name_col].astype(str).map(normalise_med_name)
    work["_qty"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0)

    work = work[work["_name"].astype(str).str.strip() != ""]
    summary = (
        work.groupby("_name", as_index=False)["_qty"]
            .sum()
            .rename(columns={"_name": "Generic Name", "_qty": "Total Qty"})
            .sort_values("Generic Name", kind="stable")
    )
    return summary

st.title("📦 Inventory Totals by Generic (ignores APO/TEVA/SANDOZ...)")

uploaded = st.file_uploader("Upload inventory CSV", type=["csv"])

st.sidebar.header("Prefix settings")
custom_prefixes = st.sidebar.text_area(
    "Add extra prefixes (one per line)",
    value="",
    help="Example:\nAPO\nTEVA\nSANDOZ\nAURO"
)

# Merge custom prefixes
if custom_prefixes.strip():
    for p in custom_prefixes.splitlines():
        p = p.strip().upper()
        if p:
            PREFIXES.add(p)

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    try:
        auto_name, auto_qty = detect_columns(df)
    except Exception:
        auto_name, auto_qty = (df.columns[0], df.columns[-1])

    st.subheader("Columns")
    col1, col2 = st.columns(2)
    with col1:
        name_col = st.selectbox("Medication name column", options=list(df.columns), index=list(df.columns).index(auto_name))
    with col2:
        qty_col = st.selectbox("Quantity column", options=list(df.columns), index=list(df.columns).index(auto_qty))

    summary = summarise(df, name_col, qty_col)

    st.subheader("Totals")
    st.caption(f"Unique generics: {len(summary)} | Sum of quantities: {summary['Total Qty'].sum():.2f}")
    st.dataframe(summary, use_container_width=True)

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary CSV",
        data=csv_bytes,
        file_name="generic_totals.csv",
        mime="text/csv"
    )
else:
    st.info("Upload a CSV to get totals.")
