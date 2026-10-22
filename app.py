import io
import math
import time
import pandas as pd
import numpy as np
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DID Requirement Report",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DARK / PURPLE THEME (CSS)
# ============================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e6e6fa;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(6px);
    }
    div[data-testid="stMetric"] label { color: #c4b5fd !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.9rem !important; font-weight: 700 !important;
    }

    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(168, 85, 247, 0.5);
        border-radius: 16px; padding: 10px;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: #ffffff; border: none; border-radius: 12px;
        padding: 0.55rem 1.4rem; font-weight: 600;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(168, 85, 247, 0.45);
    }

    div[data-testid="stDataEditor"] {
        border-radius: 14px; overflow: hidden;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1533 0%, #221c44 100%); }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 14px;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }
</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
CALLS_PER_DID = 40
EXPECTED_COL = "phone_number"

# ============================================================
# HELPERS  (all vectorized — safe for 1M+ rows)
# ============================================================
def parse_lookup(uploaded_excel) -> pd.DataFrame:
    """Read the AreaCodeandState.xlsx uploaded via the GUI."""
    df = pd.read_excel(uploaded_excel, dtype={"Areacode": str})
    df.columns = [c.strip() for c in df.columns]
    need = {"Areacode", "State", "TimeZone"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            f"Lookup file is missing required column(s): {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    df["Areacode"] = df["Areacode"].astype(str).str.strip()
    return df.dropna(subset=["Areacode", "State"])


def clean_phone_series(s: pd.Series) -> pd.Series:
    """
    Vectorized phone normalization -> clean 10-digit string or NaN.

    Handles: spaces, dashes, brackets/parentheses, dots, leading '+',
    and a '+1' / '1' country-code prefix. Runs fully in native pandas
    string ops (no Python-level loop), so it scales to millions of rows.
    """
    s = s.fillna("").astype(str).str.strip().str.lstrip("+")
    digits = s.str.replace(r"\D", "", regex=True)          # strip spaces/dashes/()/[]/dots
    is_11 = digits.str.len().eq(11) & digits.str.startswith("1")
    digits = digits.where(~is_11, digits.str[1:])          # drop US country code
    valid = digits.str.len().eq(10) & ~digits.str[0].isin(["0", "1"])
    return digits.where(valid)                              # NaN = invalid number


def read_calls_csv(file_bytes: bytes) -> pd.DataFrame:
    """Fast CSV read; tolerant of malformed lines."""
    return pd.read_csv(
        io.BytesIO(file_bytes), dtype=str,
        on_bad_lines="skip", low_memory=False, encoding_errors="ignore",
    )


def parse_calls_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read one uploaded CSV -> frame with a normalized phone + __source__ col."""
    up = read_calls_csv(file_bytes)
    col_map = {c.strip().lower(): c for c in up.columns}
    if EXPECTED_COL not in col_map:
        raise ValueError(f"'{filename}' has no '{EXPECTED_COL}' column. Found: {list(up.columns)}")
    out = up[[col_map[EXPECTED_COL]]].copy()
    out.columns = ["phone_number"]
    out["__source__"] = filename
    return out


def build_report(df_lookup: pd.DataFrame, all_calls: pd.DataFrame):
    """
    Fully vectorized pipeline. Returns (report_df, summary, unmatched_df).
    Safe for 1M+ rows: one regex pass + one join, no per-row Python.
    """
    phones = clean_phone_series(all_calls["phone_number"])
    invalid_count = int(phones.isna().sum())
    valid = phones.dropna()

    area_codes = valid.str[:3]                                   # first 3 digits
    lookup = df_lookup.set_index("Areacode")
    merged = area_codes.to_frame(name="Areacode").join(lookup, on="Areacode", how="left")

    matched = merged["State"].notna()
    unmatched_df = (
        merged.loc[~matched, "Areacode"].value_counts()
        .rename_axis("Area Code").reset_index(name="Count").sort_values("Area Code")
    )
    unmatched_count = int((~matched).sum())

    grouped = (
        merged.loc[matched].groupby("State")["Areacode"].count()
        .rename("Call Count").reset_index().sort_values("State").reset_index(drop=True)
    )
    grouped["DIDs Required"] = (grouped["Call Count"] / CALLS_PER_DID).round(3)
    grouped["DID Needed"] = grouped["DIDs Required"].apply(math.ceil)

    summary = {
        "total_states": len(grouped),
        "total_calls": int(grouped["Call Count"].sum()),
        "total_dids_required": round(float(grouped["DIDs Required"].sum()), 3),
        "total_did_needed": int(grouped["DID Needed"].sum()),
        "invalid_count": invalid_count,
        "unmatched_count": unmatched_count,
    }
    return grouped, summary, unmatched_df


# ---------- CACHED WRAPPERS (skip re-processing on every widget edit) ----------
@st.cache_data(show_spinner=False)
def cached_lookup(excel_bytes: bytes) -> pd.DataFrame:
    return parse_lookup(io.BytesIO(excel_bytes))


@st.cache_data(show_spinner=False)
def cached_calls(csv_bytes: bytes, filename: str) -> pd.DataFrame:
    return parse_calls_csv(csv_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_report(lookup_bytes: bytes, files_key: tuple) -> tuple:
    """
    files_key: tuple of (filename, bytes) — pandas hashing is fast on bytes.
    Returns (report_df, summary, unmatched_df, per_file_df) so a widget edit
    (e.g. typing in the table) never triggers a re-read of million-row files.
    """
    lookup_df = cached_lookup(lookup_bytes)
    frames, errors, file_rows = [], [], []
    for name, b in files_key:
        try:
            one = cached_calls(b, name)
        except ValueError as e:
            errors.append(str(e))
            continue
        frames.append(one)
        _, f_sum, f_un = build_report(lookup_df, one)
        file_rows.append({
            "File": name,
            "Total Numbers": len(one),
            "Invalid": f_sum["invalid_count"],
            "Matched Calls": f_sum["total_calls"],
            "Unmatched Calls": f_sum["unmatched_count"],
            "Unmatched Codes": len(f_un),
            "States": f_sum["total_states"],
            "DIDs Required": f_sum["total_dids_required"],
        })

    per_file_df = pd.DataFrame(file_rows) if file_rows else pd.DataFrame(
        columns=["File","Total Numbers","Invalid","Matched Calls",
                 "Unmatched Calls","Unmatched Codes","States","DIDs Required"]
    )
    if not frames:
        empty = pd.DataFrame(columns=["State","Call Count","DIDs Required","DID Needed"])
        return empty, None, pd.DataFrame(columns=["Area Code","Count"]), per_file_df

    all_calls = pd.concat(frames, ignore_index=True)
    report_df, summary, unmatched_df = build_report(lookup_df, all_calls)
    return report_df, summary, unmatched_df, per_file_df


def make_excel_bytes(df: pd.DataFrame, totals: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "DID Report"
    purple_fill = PatternFill("solid", fgColor="6D28D9")
    total_fill = PatternFill("solid", fgColor="3B0764")
    white_bold = Font(color="FFFFFF", bold=True)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        ws.append(row)
        for cell in ws[r_idx]:
            if r_idx == 1:
                cell.fill, cell.font = purple_fill, white_bold
                cell.alignment = Alignment(horizontal="center")
    ws.append(["TOTAL", totals["total_calls"], totals["total_dids_required"], totals["total_did_needed"]])
    for cell in ws[ws.max_row]:
        cell.fill, cell.font = total_fill, white_bold
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = (
            max(len(str(c.value)) if c.value is not None else 0 for c in col) + 4)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def style_total_row(df: pd.DataFrame, total_row_index: int):
    return df.style.apply(
        lambda r: ["background-color: #3B0764; color: white; font-weight: 700;"] * len(r)
        if r.name == total_row_index else [""] * len(r),
        axis=1,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📞 DID Report Tool")
    st.caption("State-wise DID requirement analysis")
    st.divider()
    st.markdown(
        """
**How it works**
1. Upload the area-code lookup **Excel**
2. Upload **one or more** call-data CSVs
3. Area codes matched against the lookup
4. DIDs Required = calls ÷ 40
5. Edit **DID Needed** inline, then download
        """
    )
    st.divider()
    st.caption("⚡ Optimized for 1M+ row files (vectorized pandas + caching).")

# ============================================================
# HEADER
# ============================================================
st.title("State-wise DID Requirement Report")
st.caption("Upload your lookup table and call-data CSV(s) to generate the report automatically.")

# ============================================================
# STEP 1 — LOOKUP EXCEL
# ============================================================
st.subheader("1️⃣ Area Code Mapping (Excel)")
lookup_file = st.file_uploader(
    "Upload AreaCodeandState.xlsx  (columns: Areacode, State, TimeZone)",
    type=["xlsx", "xls"], key="lookup_uploader",
    help="This file acts as the area-code → state lookup table.",
)

if lookup_file is None:
    st.info("👆 Upload the area-code mapping Excel to continue.")
    with st.expander("📄 Expected lookup format"):
        st.code("Areacode\tState\tTimeZone\n602\tAZ\tMST\n303\tCO\tMST\n410\tMD\tEST", language="text")
    st.stop()

try:
    lookup_df = cached_lookup(lookup_file.getvalue())
except ValueError as e:
    st.error(f"⚠️ {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Could not read the lookup file: {e}")
    st.stop()

st.success(f"✅ Lookup loaded: **{len(lookup_df)}** area codes from `{lookup_file.name}`")

# ============================================================
# STEP 2 — ONE OR MORE CALL CSVs
# ============================================================
st.subheader("2️⃣ Call Data (CSV — multiple files allowed)")
csv_files = st.file_uploader(
    "Upload one or more CSV files  (each must contain a `phone_number` column)",
    type=["csv"], accept_multiple_files=True, key="csv_uploader",
    help="Ctrl/Cmd-click to select several files. Records from all files are combined.",
)

if not csv_files:
    st.info("👆 Upload at least one CSV file to generate the report.")
    with st.expander("📄 Expected CSV format"):
        st.code("phone_number\n6021234567\n3035551234\n4109876543", language="csv")
    st.stop()

# ============================================================
# PROCESS (cached — runs once per unique file set)
# ============================================================
files_key = tuple((f.name, f.getvalue()) for f in csv_files)

t0 = time.perf_counter()
with st.spinner(f"Processing {len(csv_files)} file(s)…"):
    report_df, summary, unmatched_df, per_file_df = cached_report(lookup_file.getvalue(), files_key)
elapsed = time.perf_counter() - t0

if summary is None:
    st.error("No usable call records found in the uploaded file(s).")
    st.stop()

# ---------- PER-FILE SUMMARY (single table + TOTAL row) ----------
pf_total = pd.DataFrame([{
    "File": "TOTAL",
    "Total Numbers": int(per_file_df["Total Numbers"].sum()),
    "Invalid": int(per_file_df["Invalid"].sum()),
    "Matched Calls": int(per_file_df["Matched Calls"].sum()),
    "Unmatched Calls": int(per_file_df["Unmatched Calls"].sum()),
    "Unmatched Codes": int(per_file_df["Unmatched Codes"].sum()),
    "States": int(per_file_df["States"].sum()),
    "DIDs Required": round(float(per_file_df["DIDs Required"].sum()), 3),
}])
pf_view = pd.concat([per_file_df, pf_total], ignore_index=True)
pf_view["DIDs Required"] = pf_view["DIDs Required"].round(3)
st.subheader("📁 Per-File Summary")
st.caption(f"One row per uploaded CSV, plus a TOTAL row. ⏱ Processed in {elapsed:.2f}s "
           f"(cached — instant on re-runs).")
st.dataframe(style_total_row(pf_view, len(per_file_df)), use_container_width=True, hide_index=True)
st.divider()

# ---------- SUMMARY CARDS ----------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("States", summary["total_states"])
c2.metric("Total Calls", f"{summary['total_calls']:,}")
c3.metric("DIDs Required", f"{summary['total_dids_required']:,.3f}".rstrip("0").rstrip("."))
c4.metric("DID Needed", f"{summary['total_did_needed']:,}")
c5.metric("Invalid Numbers", summary["invalid_count"])
c6.metric("Unmatched Area Codes", summary["unmatched_count"])

st.divider()

if report_df.empty:
    st.warning("No phone numbers could be matched to a state.")
else:
    # ============================================================
    # REPORT SECTION — TWO TABLES
    # ============================================================
    st.subheader("📊 Report")

    # ---- TABLE 1: editable report ----
    st.markdown("**① State-wise detail — edit `DID Needed` inline**")
    edited_df = st.data_editor(
        report_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "State": st.column_config.TextColumn("State", disabled=True),
            "Call Count": st.column_config.NumberColumn("Call Count", disabled=True, format="%d"),
            "DIDs Required": st.column_config.NumberColumn("DIDs Required", disabled=True, format="%.3f"),
            "DID Needed": st.column_config.NumberColumn(
                "DID Needed", min_value=0, step=1, format="%d",
                help="Manually adjust the DIDs to provision for this state.",
            ),
        },
        key="did_editor",
    )
    edited_df["DID Needed"] = pd.to_numeric(edited_df["DID Needed"], errors="coerce").fillna(0).astype(int)

    # live totals driven by user edits
    total_calls = int(edited_df["Call Count"].sum())
    total_req = round(float(edited_df["DIDs Required"].sum()), 3)
    total_needed = int(edited_df["DID Needed"].sum())

    # ---- TABLE 2: grand totals (fixed at bottom, updates instantly) ----
    st.markdown("**② Grand totals — reflects your `DID Needed` edits**")
    totals_df = pd.DataFrame([{
        "States": len(edited_df),
        "TOTAL Call Count": total_calls,
        "TOTAL DIDs Required": total_req,
        "TOTAL DID Needed": total_needed,
    }])
    st.dataframe(
        totals_df.style.apply(
            lambda r: ["background-color: #3B0764; color: white; font-weight: 700;"] * len(r),
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------- UNMATCHED AREA CODES ----------
if not unmatched_df.empty:
    with st.expander(
        f"🔍 Unmatched area codes — {summary['unmatched_count']} calls "
        f"({len(unmatched_df)} distinct codes)"
    ):
        st.dataframe(unmatched_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download unmatched codes (CSV)",
            unmatched_df.to_csv(index=False).encode("utf-8"),
            "unmatched_area_codes.csv", "text/csv",
        )
else:
    st.success("✅ Every area code matched the lookup table.")

# ---------- DOWNLOADS ----------
st.divider()
st.subheader("⬇️ Export")

if report_df.empty:
    export_df = pd.DataFrame(columns=["State","Call Count","DIDs Required","DID Needed"])
    totals_export = {"total_calls": 0, "total_dids_required": 0.0, "total_did_needed": 0}
else:
    totals_export = {"total_calls": total_calls,
                     "total_dids_required": total_req, "total_did_needed": total_needed}
    total_row = pd.DataFrame([{"State": "TOTAL", **{k.replace("total_", ""): v for k, v in [
        ("total_calls", total_calls),
        ("total_dids_required", total_req),
        ("total_did_needed", total_needed)]}}])
    export_df = pd.concat([edited_df, total_row], ignore_index=True)

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
xlsx_bytes = make_excel_bytes(edited_df if not report_df.empty else export_df, totals_export)

d1, d2 = st.columns(2)
with d1:
    st.download_button("Download as CSV", csv_bytes,
        "did_requirement_report.csv", "text/csv", use_container_width=True)
with d2:
    st.download_button("Download as Excel", xlsx_bytes,
        "did_requirement_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
