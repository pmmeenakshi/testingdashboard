# === Bintix Waste Analytics — Optimized (CSV/Parquet) ===
# Note: Same features & UI; faster IO (Parquet), tighter caching, no NameErrors.

import re
import io
import base64
import mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# ---------------- App & Brand ----------------
st.set_page_config(page_title="Bintix Waste Analytics", layout="wide")

BRAND_PRIMARY = "#36204D"     # purple
TEXT_DARK = "#36204D"

# Speed settings
ST_MAP_HEIGHT = 900
ST_RETURNED_OBJECTS = []  # don't send all map layers back to Streamlit

# --- Environmental conversions ---
CO2_PER_KG_DRY = 2.18      # 1 kg dry waste -> 2.18 kg CO2 averted
KG_PER_TREE     = 117.0     # 117 kg dry waste -> 1 tree saved

# ---------------- Assets (icons) ----------------
BASE_DIR = Path(__file__).parent.resolve()
_ASSET_DIR_CANDIDATES = [BASE_DIR / "assets", BASE_DIR / "assests"]
ASSETS_DIR = next((p for p in _ASSET_DIR_CANDIDATES if p.exists()), _ASSET_DIR_CANDIDATES[0])

@st.cache_resource(show_spinner=False)
def load_icon_data_uri(filename: str) -> str:
    """Return a data: URI for an image in ASSETS_DIR so it renders inside Folium popups."""
    p = ASSETS_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Icon not found: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

try:
    TREE_ICON     = load_icon_data_uri("tree.png")
    HOUSE_ICON    = load_icon_data_uri("house.png")
    RECYCLE_ICON  = load_icon_data_uri("waste-management.png")
except FileNotFoundError as e:
    st.error(f"{e}\nMake sure your icons are in: {ASSETS_DIR}")
    TREE_ICON = HOUSE_ICON = RECYCLE_ICON = ""

# ---------------- Data loading (CSV/Parquet) ----------------
CSV_DEFAULT = BASE_DIR / "standardized_wide_fy2024_25.csv"
PARQUET_DEFAULT = BASE_DIR / "standardized_wide_fy2024_25.parquet"

ID_COLS_REQUIRED = ["City", "Community", "Pincode"]
ID_COLS_OPTIONAL = ["Lat", "Lon"]
METRIC_COL_REGEX = re.compile(
    r"^(Impact|Tonnage|CO2_Kgs_Averted|Households_Participating|Segregation_Compliance_Pct)_(\d{4}-\d{2})$"
)

def _detect_metric_month_cols(columns):
    cols, months = [], set()
    for c in columns:
        m = METRIC_COL_REGEX.match(c)
        if m:
            cols.append(c); months.add(m.group(2))
    return cols, sorted(months)

def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet based on suffix (fast path prefers parquet)."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    # default CSV
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_and_prepare_auto(default_csv: Path, default_parquet: Path):
    """
    Load the wide table from:
      1) default_parquet if exists, else
      2) default_csv
    Prepare long format and months.
    """
    if default_parquet.exists():
        src = default_parquet
    elif default_csv.exists():
        src = default_csv
    else:
        raise FileNotFoundError(f"No data file found. Expected one of:\n{default_parquet}\n{default_csv}")

    df = _read_table(src)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric-month columns like Impact_2024-04, Tonnage_2025-03 found.")

    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"]).sort_values(id_cols_present + ["Metric", "Date"])
    return df, long_df, months, str(src.resolve())

# ---------------- Sidebar (upload) ----------------
with st.sidebar:
    uploaded = st.file_uploader(
        "Upload dataset (CSV or Parquet)",
        type=["csv", "parquet"],
        help="Wide format: one row per community; monthly cols like Impact_2024-04, Tonnage_2024-04, ..."
    )
    st.caption("If no upload, the default data file from the app folder is used (prefers Parquet if present).")
    show_popup_charts = st.toggle(
        "Show charts in popups (slower)",
        value=False,
        help="Renders mini charts inside each popup. Turn off for fast map updates.",
        key="toggle_popup_charts"     # ← add this
    )

@st.cache_data(show_spinner=False)
def load_uploaded(file) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Load uploaded CSV/Parquet with same prep as defaults."""
    name = file.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(file)
        src = f"uploaded: {file.name}"
    else:
        df = pd.read_csv(file)
        src = f"uploaded: {file.name}"

    df = df.rename(columns={c: c.strip() for c in df.columns})

    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric-month columns like Impact_YYYY-MM, Tonnage_YYYY-MM found.")

    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"]).sort_values(id_cols_present + ["Metric", "Date"])
    return df, long_df, months, src

# ---------------- Initial load (defaults or upload) ----------------
try:
    if uploaded is not None:
        df_wide, df_long, months, data_src = load_uploaded(uploaded)
    else:
        df_wide, df_long, months, data_src = load_and_prepare_auto(CSV_DEFAULT, PARQUET_DEFAULT)

    st.session_state["df_wide"] = df_wide
    st.session_state["df_long"] = df_long
    st.session_state["months"]  = months
    st.session_state["data_src"] = data_src
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------------- Minor UI theming fix (unchanged features) ----------------
st.markdown(
    """
    <style>
    div[data-baseweb="select"] { background-color: #FFFFFF !important; }
    div[data-baseweb="select"] span { color: white !important; }
    div[data-baseweb="select"] input { color: #36204D !important; }
    div[data-baseweb="menu"] { background-color: #FFFFFF !important; }
    div[data-baseweb="menu"] div[role="option"] { color: white !important; }
    div[data-baseweb="menu"] div[role="option"]:hover { background-color: #36204D22 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Data from session ----------------
df_wide  = st.session_state["df_wide"]
df_long  = st.session_state["df_long"]
months   = st.session_state["months"]
data_src = st.session_state["data_src"]

# Normalize key id columns to STRING
for col in ["Pincode", "Community", "City"]:
    if col in df_wide.columns:
        df_wide[col] = df_wide[col].astype(str)
    if col in df_long.columns:
        df_long[col] = df_long[col].astype(str)

# Merge pincode centroids if needed
PINCODE_LOOKUP = BASE_DIR / "pincode_centroids.csv"
if ("Lat" not in df_wide.columns or "Lon" not in df_wide.columns):
    if PINCODE_LOOKUP.exists():
        try:
            look = pd.read_csv(PINCODE_LOOKUP)
            look.columns = [c.strip() for c in look.columns]
            if {"Pincode","Lat","Lon"}.issubset(look.columns):
                look["Pincode"] = look["Pincode"].astype(str).str.strip()
                look["Lat"] = pd.to_numeric(look["Lat"], errors="coerce")
                look["Lon"] = pd.to_numeric(look["Lon"], errors="coerce")
                df_wide["Pincode"] = df_wide["Pincode"].astype(str).str.strip()
                df_wide = df_wide.merge(look[["Pincode","Lat","Lon"]], on="Pincode", how="left")
                st.caption(
                    f"🗺️ Coordinates merged from `pincode_centroids.csv` "
                    f"(markers available for {(df_wide[['Lat','Lon']].notna().all(axis=1)).sum()} communities)."
                )
            else:
                st.warning("`pincode_centroids.csv` columns must be exactly: Pincode, Lat, Lon.")
        except Exception as e:
            st.warning(f"Could not read/merge `pincode_centroids.csv`: {e}")

# Persist normalized/merged
st.session_state["df_wide"] = df_wide
st.session_state["df_long"] = df_long

# ---------------- Title ----------------
st.markdown("<h1>Smart Waste Analytics — FY 2024–25</h1>", unsafe_allow_html=True)
st.caption(f"Data source: {data_src}")

# ---------------- Global Filters ----------------
st.markdown("### 🔎 Global Filters")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.2])

city_opts = sorted(df_wide["City"].dropna().unique().tolist()) if "City" in df_wide else []
comm_opts = sorted(df_wide["Community"].dropna().unique().tolist()) if "Community" in df_wide else []
pin_opts  = sorted(df_wide["Pincode"].dropna().unique().tolist()) if "Pincode" in df_wide else []

with c1: sel_city = st.multiselect("City", city_opts, placeholder="All")
with c2: sel_comm = st.multiselect("Community", comm_opts, placeholder="All")
with c3: sel_pin  = st.multiselect("Pincode", pin_opts,  placeholder="All")
with c4:
    start_m, end_m = st.select_slider("Date range (month)", options=months, value=(months[0], months[-1]))

def apply_filters(dfw, dfl):
    dfw = dfw.copy(); dfl = dfl.copy()
    for col in ["Pincode", "Community", "City"]:
        if col in dfw: dfw[col] = dfw[col].astype(str)
        if col in dfl: dfl[col] = dfl[col].astype(str)

    sel_city_s = [str(x) for x in sel_city] if sel_city else []
    sel_comm_s = [str(x) for x in sel_comm] if sel_comm else []
    sel_pin_s  = [str(x) for x in sel_pin]  if sel_pin  else []

    mask_w = pd.Series(True, index=dfw.index)
    if sel_city_s: mask_w &= dfw["City"].isin(sel_city_s)
    if sel_comm_s: mask_w &= dfw["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_w &= dfw["Pincode"].isin(sel_pin_s)
    dfw_f = dfw[mask_w].copy()

    mask_l = pd.Series(True, index=dfl.index)
    if sel_city_s: mask_l &= dfl["City"].isin(sel_city_s)
    if sel_comm_s: mask_l &= dfl["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_l &= dfl["Pincode"].isin(sel_pin_s)
    d0 = pd.to_datetime(start_m + "-01"); d1 = pd.to_datetime(end_m + "-01")
    mask_l &= (dfl["Date"] >= d0) & (dfl["Date"] <= d1)
    dfl_f = dfl[mask_l].copy()
    return dfw_f, dfl_f

dfw_filt, dfl_filt = apply_filters(df_wide, df_long)

# ---------------- Summary KPIs ----------------
def kpi_value(dfl, metric, agg="sum"):
    s = dfl.loc[dfl["Metric"] == metric, "Value"]
    if s.empty: return 0.0
    return float(s.sum() if agg == "sum" else s.mean())

n_communities = dfw_filt["Community"].nunique() if "Community" in dfw_filt else 0
n_cities      = dfw_filt["City"].nunique()      if "City" in dfw_filt else 0
total_tonnage = kpi_value(dfl_filt, "Tonnage", "sum")
total_co2     = kpi_value(dfl_filt, "CO2_Kgs_Averted", "sum")
avg_comp      = kpi_value(dfl_filt, "Segregation_Compliance_Pct", "mean")
total_hh      = kpi_value(dfl_filt, "Households_Participating", "sum")

st.markdown("### 📊 Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
k1.metric("Communities", n_communities)
k2.metric("Cities", n_cities)
k3.metric("Total Tonnage", f"{total_tonnage:,.0f}")
k4.metric("CO₂ Averted (kg)", f"{total_co2:,.0f}")
k5.metric("Avg Segregation (%)", f"{avg_comp:,.1f}")
k6.metric("Active Households", f"{total_hh:,.0f}")
st.caption(f"Period: **{start_m} → {end_m}**")

# ---------------- Tabs ----------------
tab_map, tab_insights = st.tabs(["🗺️ 2D Map & Popups", "🧠 Insights"])

# --- Trends helper (kept for completeness) ---
def small_trend(df_long_or_filtered, community_id, metric):
    d = df_long_or_filtered[
        (df_long_or_filtered["Community"] == str(community_id)) &
        (df_long_or_filtered["Metric"] == metric)
    ].sort_values("Date")
    if d.empty:
        return None
    fig = px.line(
        d, x="Date", y="Value", markers=True,
        labels={"Value": metric.replace("_"," "), "Date": "Date"},
        template=None
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=36, b=36),
        title=dict(text=f"{metric.replace('_',' ')} Trend",
                   font=dict(color=TEXT_DARK, size=16)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Poppins", color=TEXT_DARK, size=13),
        xaxis=dict(color=TEXT_DARK, gridcolor="rgba(54,32,77,0.12)", zerolinecolor="rgba(54,32,77,0.18)", linecolor="rgba(54,32,77,0.25)"),
        yaxis=dict(color=TEXT_DARK, gridcolor="rgba(54,32,77,0.12)", zerolinecolor="rgba(54,32,77,0.18)", linecolor="rgba(54,32,77,0.25)"),
        showlegend=False,
    )
    fig.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                      marker=dict(color=BRAND_PRIMARY))
    return fig

def _to_data_uri(fig, w=340):
    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=180)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' style='width:{w}px;height:auto;border:0;'/>"

def _distinct_colors(n):
    cmaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1]
    colors = []
    i = 0
    while len(colors) < n:
        cmap = cmaps[i % len(cmaps)]
        M = cmap.N
        take = min(n - len(colors), M)
        for j in range(take):
            colors.append(cmap(j / max(M - 1, 1)))
        i += 1
    return colors[:n]

# --- DATE-RANGE AWARE POPUP SUMMARY ---
@st.cache_data(show_spinner=False)
def summarize_for_popup(dfl_filtered: pd.DataFrame, community_id: str, pincode: str|None):
    """
    Returns popup KPIs based ONLY on current date-filtered data.
    Keys: tonnage, co2, households, seg_pct, trees
    """
    d = dfl_filtered.copy()
    d["Community"] = d["Community"].astype(str)
    if pincode is not None and "Pincode" in d.columns:
        d["Pincode"] = d["Pincode"].astype(str)

    d = d[d["Community"] == str(community_id)]
    if pincode is not None:
        d = d[d["Pincode"] == str(pincode)]

    def agg(metric: str, how="sum") -> float:
        s = d.loc[d["Metric"] == metric, "Value"]
        if s.empty:
            return 0.0
        return float(s.sum() if how == "sum" else s.mean())

    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_kg = 0.0
    for m in dry_candidates:
        s = d.loc[d["Metric"] == m, "Value"]
        if not s.empty:
            dry_kg = float(s.sum())
            break

    co2 = dry_kg * CO2_PER_KG_DRY
    trees = dry_kg / KG_PER_TREE

    return {
        "tonnage":    agg("Tonnage", "sum"),
        "co2":        co2,
        "households": agg("Households_Participating", "sum"),
        "seg_pct":    agg("Segregation_Compliance_Pct", "mean"),
        "trees":      trees,
    }

@st.cache_data(show_spinner=False)
def monthly_series(df_long, community: str, metric: str):
    d = df_long[
        (df_long["Community"].astype(str) == str(community)) &
        (df_long["Metric"] == metric)
    ][["Date", "Value"]].sort_values("Date").copy()
    return d

@st.cache_data(show_spinner=False)
def popup_charts_for_comm(dfl_filtered: pd.DataFrame, community_id: str):
    """
    Cached chart generator for popup.
    Uses date-filtered data (dfl_filtered) for speed and accuracy.
    Returns (bar_img, donut_img) HTML <img> tags (data URIs).
    """
    BRAND = BRAND_PRIMARY
    dm = dfl_filtered.copy()
    dm["Community"] = dm["Community"].astype(str)
    dm = dm[dm["Community"] == str(community_id)]
    if dm.empty:
        return "", ""

    dm["MonthKey"] = dm["Date"].dt.to_period("M")

    # TONNAGE as line chart
    bar_img = ""
    d_ton = dm[dm["Metric"] == "Tonnage"][["MonthKey", "Value"]]
    if not d_ton.empty:
        d_ton = d_ton.groupby("MonthKey", as_index=False)["Value"].sum().sort_values("MonthKey")
        xlab = [p.to_timestamp().strftime("%b") for p in d_ton["MonthKey"]]
        fig, ax = plt.subplots(figsize=(3.1, 1.6), dpi=120)
        ax.plot(xlab, d_ton["Value"], marker="o", lw=1.6, color=BRAND)
        ax.set_title("Tonnage", fontsize=9, color=BRAND, pad=2)
        ax.tick_params(axis="x", labelsize=8, colors=BRAND)
        ax.tick_params(axis="y", labelsize=8, colors=BRAND)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(alpha=0.12, axis="y")
        bar_img = _to_data_uri(fig, w=300)

    # CO2 donut
    donut_img = ""
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_month = None
    for m in dry_candidates:
        cur = dm[dm["Metric"] == m][["MonthKey", "Value"]]
        if not cur.empty:
            dry_month = cur
            break
    if dry_month is not None:
        d = dry_month.groupby("MonthKey", as_index=False)["Value"].sum().sort_values("MonthKey")
        vals = (d["Value"] * CO2_PER_KG_DRY).clip(lower=0.0).to_numpy()
        labels = [p.to_timestamp().strftime("%b") for p in d["MonthKey"]]
        colors = _distinct_colors(len(labels))

        fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=120)
        wedges, _ = ax.pie(vals, wedgeprops=dict(width=0.45), startangle=90, colors=colors)
        ax.set(aspect="equal")
        ax.text(0, 0, "CO₂\nAverted", ha="center", va="center",
                fontsize=9, color=BRAND, fontweight="bold", linespacing=1.1)
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, frameon=False)
        donut_img = _to_data_uri(fig, w=220)

    return bar_img, donut_img

def jitter_duplicates(df, lat_col="Lat", lon_col="Lon", jitter_deg=0.00025):
    """
    Move markers that share the same (Lat, Lon) into a tiny circle around
    the original location so each one gets a working tooltip/popup.
    jitter_deg ~ 0.00025 ≈ 25–30 meters.
    """
    df = df.copy()
    gb = df.groupby([df[lat_col].round(6), df[lon_col].round(6)])
    for _, idx in gb.groups.items():
        n = len(idx)
        if n > 1:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = jitter_deg
            df.loc[idx, lat_col] = df.loc[idx, lat_col].to_numpy() + r * np.sin(angles)
            df.loc[idx, lon_col] = df.loc[idx, lon_col].to_numpy() + r * np.cos(angles)
    return df

# ---------------- Map Tab ----------------
with tab_map:
    has_latlon = (
        "Lat" in dfw_filt.columns and "Lon" in dfw_filt.columns and
        dfw_filt[["Lat","Lon"]].notna().all(axis=1).any()
    )

    if not has_latlon:
        st.warning("Map needs coordinates. Add **Lat/Lon** columns or merge a `pincode_centroids.csv`.")
        st.info("Click markers to see details here (after coordinates are available).")
        selected_comm, selected_pin = None, None
    else:
        valid = dfw_filt.dropna(subset=["Lat", "Lon"])
        valid = jitter_duplicates(valid)

        lat0 = float(valid["Lat"].mean())
        lon0 = float(valid["Lon"].mean())
        fmap = folium.Map(location=[lat0, lon0], zoom_start=11, tiles="cartodbpositron")

        cluster = MarkerCluster().add_to(fmap)

        comm_arr = valid["Community"].astype(str).to_numpy()
        pin_arr  = valid["Pincode"].astype(str).to_numpy()
        lat_arr  = valid["Lat"].astype(float).to_numpy()
        lon_arr  = valid["Lon"].astype(float).to_numpy()
        city_arr = valid["City"].astype(str).to_numpy() if "City" in valid else np.array([""]*len(valid))

        for comm, pin, lat, lon, city in zip(comm_arr, pin_arr, lat_arr, lon_arr, city_arr):
            stats = summarize_for_popup(dfl_filt, community_id=comm, pincode=pin)

            bar_img, donut_img = "", ""
            if show_popup_charts:
                try:
                    bar_img, donut_img = popup_charts_for_comm(dfl_filt, comm)
                except Exception:
                    bar_img, donut_img = "", ""

            popup_html = f"""
            <div style='font-family:Poppins; width:360px;'>
                <h4 style='margin:0 0 4px 0; color:#36204D;'>{comm}</h4>
                <div style='font-size:12px; color:#333;'>City: {city} | Pincode: {pin}</div>
                <hr style='margin:6px 0;'>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{TREE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['trees']:,.0f} Trees Saved</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{HOUSE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['households']:,.0f} Households Participating</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{RECYCLE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['seg_pct']:.1f}% Segregation</b></span>
                </div>
                <hr style='margin:8px 0;'>
                <div style='margin-bottom:8px;'>
                    <b>CO₂ Averted</b>
                    {donut_img}
                </div>
                <div style='margin-top:6px;'>
                    <b>Tonnage</b>
                    {bar_img}
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=BRAND_PRIMARY,
                fill=True,
                fill_color=BRAND_PRIMARY,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(f"{comm} • {pin}"),
                popup=folium.Popup(popup_html, max_width=420),
            ).add_to(cluster)

        st.markdown("##### Map")
        map_event = st_folium(
            fmap,
            height=ST_MAP_HEIGHT,
            use_container_width=True,
            returned_objects=ST_RETURNED_OBJECTS
        )

        selected_comm, selected_pin = None, None
        if map_event and map_event.get("last_object_clicked_tooltip"):
            tip = map_event["last_object_clicked_tooltip"]  # "COMMUNITY • PINCODE"
            parts = [p.strip() for p in tip.split("•")]
            if len(parts) == 2:
                selected_comm, selected_pin = parts[0], parts[1]
            else:
                selected_comm = parts[0]
        elif not dfw_filt.empty:
            selected_comm = str(dfw_filt.iloc[0]["Community"])
            selected_pin  = str(dfw_filt.iloc[0]["Pincode"])

    # --- KPIs for selected community ---
    cA, cB, cC, cD = st.columns(4)
    stats = summarize_for_popup(dfl_filt, community_id=selected_comm, pincode=selected_pin)
    cA.metric("Tonnage (kg)", f"{stats['tonnage']:,.0f}")
    cB.metric("CO₂ Averted (kg)", f"{stats['co2']:,.0f}")
    cC.metric("Households", f"{stats['households']:,.0f}")
    cD.metric("Segregation % (avg)", f"{stats['seg_pct']:.1f}")

    # --- Trends for selected community ---
    st.markdown("#### Trends (Selected Community)")

    ton = monthly_series(dfl_filt, selected_comm, "Tonnage")
    if not ton.empty:
        fig_ton = px.line(
            ton, x="Date", y="Value",
            title="Tonnage over Time",
            labels={"Value": "Tonnage (kg)", "Date": "Date"},
            markers=True,
        )
        fig_ton.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY))
        fig_ton.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="Tonnage (kg)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
        )
        st.plotly_chart(fig_ton, use_container_width=True)
    else:
        st.info("No tonnage data in this date range for the selected community.")

    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry = None
    for m in dry_candidates:
        s = monthly_series(dfl_filt, selected_comm, m)
        if not s.empty:
            dry = s
            break

    if dry is not None and not dry.empty:
        co2 = dry.copy()
        co2["CO2_kg"] = (co2["Value"] * CO2_PER_KG_DRY).clip(lower=0.0)
        fig_co2 = px.line(
            co2, x="Date", y="CO2_kg",
            markers=True,
            title="CO₂ Averted (Calculated) over Time",
            labels={"CO2_kg": "CO₂ Averted (kg)", "Date": "Date"},
        )
        fig_co2.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY, size=6))
        fig_co2.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="CO₂ Averted (kg)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            showlegend=False,
        )
        st.plotly_chart(fig_co2, use_container_width=True)
    else:
        st.info("No dry/tonnage series available to compute CO₂ for this community.")

    seg = monthly_series(dfl_filt, selected_comm, "Segregation_Compliance_Pct")
    if not seg.empty:
        fig_seg = px.line(
            seg, x="Date", y="Value",
            markers=True,
            title="Segregation % over Time",
            labels={"Value": "Segregation (%)", "Date": "Date"},
        )
        fig_seg.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                              marker=dict(color=BRAND_PRIMARY, size=6))
        fig_seg.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(title=dict(text="Date", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            yaxis=dict(title=dict(text="Segregation (%)", font=dict(color="#000")), tickfont=dict(color="#000"), gridcolor="#EEE", zerolinecolor="#EEE"),
            showlegend=False,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

# ---------------- Insights Tab ----------------
with tab_insights:
    st.markdown("### 🧠 Auto Insights (All Cities in Selected Date Range)")

    d0 = pd.to_datetime(start_m + "-01")
    d1 = pd.to_datetime(end_m + "-01")
    dfl_date = df_long[(df_long["Date"] >= d0) & (df_long["Date"] <= d1)].copy()

    for col in ["City", "Community", "Pincode"]:
        if col in dfl_date.columns:
            dfl_date[col] = dfl_date[col].astype(str)

    def _brand_axes(fig, title=None):
        fig.update_layout(
            title=title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None),
            font=dict(family="Poppins", color=BRAND_PRIMARY, size=14),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
        )
        fig.update_xaxes(title_font=dict(color=BRAND_PRIMARY, size=12), tickfont=dict(color=BRAND_PRIMARY, size=11),
                         gridcolor="#EEE", zerolinecolor="#EEE")
        fig.update_yaxes(title_font=dict(color=BRAND_PRIMARY, size=12), tickfont=dict(color=BRAND_PRIMARY, size=11),
                         gridcolor="#EEE", zerolinecolor="#EEE")
        return fig

    sum_metrics  = ["Tonnage", "CO2_Kgs_Averted", "Households_Participating"]
    mean_metrics = ["Segregation_Compliance_Pct", "Impact"]

    city_sum = (
        dfl_date[dfl_date["Metric"].isin(sum_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    city_mean = (
        dfl_date[dfl_date["Metric"].isin(mean_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="mean", fill_value=0.0)
        .reset_index()
    )

    def _top_city(df, metric, how="max"):
        if df.empty or metric not in df.columns:
            return "—", 0.0
        row = df.loc[df[metric].idxmax()] if how == "max" else df.loc[df[metric].idxmin()]
        return str(row["City"]), float(row[metric])

    colA, colB, colC, colD = st.columns(4)
    t_city, t_val = _top_city(city_sum, "Tonnage", "max")
    c_city, c_val = _top_city(city_sum, "CO2_Kgs_Averted", "max")
    h_city, h_val = _top_city(city_sum, "Households_Participating", "max")
    s_city, s_val = _top_city(city_mean, "Segregation_Compliance_Pct", "max")

    with colA:
        st.caption("Top city by Tonnage")
        st.subheader(t_city)
        st.success(f"↑ {t_val:,.0f}")
    with colB:
        st.caption("Top city by CO₂ averted (kg)")
        st.subheader(c_city)
        st.success(f"↑ {c_val:,.0f}")
    with colC:
        st.caption("Top city by Households")
        st.subheader(h_city)
        st.success(f"↑ {h_val:,.0f}")
    with colD:
        st.caption("Highest Avg Segregation (%)")
        st.subheader(s_city)
        st.success(f"↑ {s_val:,.1f}%")

    st.markdown("---")

    if not city_sum.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                city_sum.sort_values("Tonnage", ascending=False),
                x="City", y="Tonnage",
                text="Tonnage", title="Total Tonnage by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c2:
            fig = px.bar(
                city_sum.sort_values("CO2_Kgs_Averted", ascending=False),
                x="City", y="CO2_Kgs_Averted",
                text="CO2_Kgs_Averted", title="CO₂ Averted (kg) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                city_sum.sort_values("Households_Participating", ascending=False),
                x="City", y="Households_Participating",
                text="Households_Participating", title="Households by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c4:
            fig = px.bar(
                city_mean.sort_values("Segregation_Compliance_Pct", ascending=False),
                x="City", y="Segregation_Compliance_Pct",
                text="Segregation_Compliance_Pct", title="Avg Segregation (%) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

    st.markdown("---")

    topN = 10
    comm_tonn = (
        dfl_date[dfl_date["Metric"] == "Tonnage"]
        .groupby(["City", "Community", "Pincode"], as_index=False)["Value"].sum()
        .rename(columns={"Value": "Tonnage"})
        .sort_values("Tonnage", ascending=False)
        .head(topN)
    )

    st.markdown(f"#### Top {topN} Communities by Tonnage (All Cities)")
    if not comm_tonn.empty:
        fig_comm = px.bar(
            comm_tonn,
            x="Community", y="Tonnage",
            color="City",
            title="Top Communities by Total Tonnage",
            labels={"Value": "Tonnage", "Community": "Community"},
        )
        fig_comm.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        fig_comm = _brand_axes(fig_comm)
        st.plotly_chart(fig_comm, use_container_width=True)
        st.caption("Tip: Hover a bar to see its city and pincode.")
    else:
        st.info("No community tonnage available in this date range.")

    st.markdown("---")

    st.write("The table below reflects the **current filters** (city/community/pincode + date range).")
    filtered_csv = dfl_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Trends (filtered CSV)",
        data=filtered_csv,
        file_name="trends_filtered.csv",
        mime="text/csv",
        key="dl_trends_bottom",
    )
    st.dataframe(dfl_filt, use_container_width=True, height=420)

























































































# === BLOCK 1: Minimal loader (Parquet preferred, CSV fallback) ===
import re
import io
import base64
import mimetypes
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import io, base64
from pathlib import Path
import time

from streamlit_folium import st_folium
import folium
import plotly.express as px
from folium.plugins import MarkerCluster
CO2_PER_KG_DRY = 2.18      # 1 kg dry waste -> 2.18 kg CO2 averted
KG_PER_TREE    = 117.0     # 117 kg dry waste -> 1 tree saved

BRAND_PRIMARY = "#36204D"     # purple
SECONDARY_GREEN = "#2E7D32"   # (kept only if you need elsewhere; not used on map)
TEXT_DARK = "#36204D" 
ST_MAP_HEIGHT = 560
ST_RETURNED_OBJECTS = []  # don't send all map layers back to Streamlit


# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.resolve()
CSV_DEFAULT   = BASE_DIR / "standardized_wide_fy2024_25.csv"
PARQUET_WIDE  = BASE_DIR / "wide.parquet"
PARQUET_LONG  = BASE_DIR / "long.parquet"

# ---------- Assets (icons) ----------
# Try both spellings so it works whether the folder is "assets" or "assests"
_ASSET_DIR_CANDIDATES = [BASE_DIR / "assets", BASE_DIR / "assests"]
ASSETS_DIR = next((p for p in _ASSET_DIR_CANDIDATES if p.exists()), _ASSET_DIR_CANDIDATES[0])

@st.cache_resource(show_spinner=False)
def load_icon_data_uri(filename: str) -> str:
    """Return a data: URI for an image in ASSETS_DIR so it renders inside Folium popups."""
    p = ASSETS_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Icon not found: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# Load once and reuse
try:
    TREE_ICON     = load_icon_data_uri("tree.png")
    HOUSE_ICON    = load_icon_data_uri("house.png")
    RECYCLE_ICON  = load_icon_data_uri("waste-management.png")
except FileNotFoundError as e:
    st.error(f"{e}\nMake sure your icons are in: {ASSETS_DIR}")
    TREE_ICON = HOUSE_ICON = RECYCLE_ICON = ""  # safe fallbacks

# ---------- Schema constants ----------
ID_COLS_REQUIRED = ["City", "Community", "Pincode"]
ID_COLS_OPTIONAL = ["Lat", "Lon"]
METRIC_COL_REGEX = re.compile(
    r"^(Impact|Tonnage|CO2_Kgs_Averted|Households_Participating|Segregation_Compliance_Pct)_(\d{4}-\d{2})$"
)

def _detect_metric_month_cols(columns):
    cols, months = [], set()
    for c in columns:
        m = METRIC_COL_REGEX.match(c)
        if m:
            cols.append(c); months.add(m.group(2))
    return cols, sorted(months)

@st.cache_data(show_spinner=False)
def _csv_to_long_wide(csv_path: Path):
    """Original CSV path -> (df_wide, df_long, months)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric-month columns like Impact_2024-04, Tonnage_2025-03 found.")

    # numeric coercion once
    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # long format
    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df.drop(columns=["Metric_Month"], inplace=True)

    # cast ids to string (helps memory & joins)
    for c in ["City", "Community", "Pincode"]:
        if c in df.columns:      df[c] = df[c].astype("string")
        if c in long_df.columns: long_df[c] = long_df[c].astype("string")

    long_df.sort_values(ID_COLS_REQUIRED + ["Metric", "Date"], inplace=True)
    return df, long_df, months

@st.cache_data(show_spinner=False)
def load_and_prepare_fast():
    """
    Prefer Parquet for speed. If missing, build from CSV and write Parquet
    so subsequent runs are much faster.
    Returns: (df_wide, df_long, months, data_src_str)
    """
    # 1) Fast path: Parquet exists
    if PARQUET_WIDE.exists() and PARQUET_LONG.exists():
        df_wide = pd.read_parquet(PARQUET_WIDE)
        df_long = pd.read_parquet(PARQUET_LONG)
        # ensure datetime dtype (Parquet should preserve, but enforce safely)
        if not pd.api.types.is_datetime64_any_dtype(df_long["Date"]):
            df_long["Date"] = pd.to_datetime(df_long["Date"])
        months = sorted(df_long["Date"].dt.strftime("%Y-%m").unique().tolist())
        return df_wide, df_long, months, f"{PARQUET_WIDE.name} / {PARQUET_LONG.name}"

    # 2) Build from CSV, then write Parquet for the next run
    df_wide, df_long, months = _csv_to_long_wide(CSV_DEFAULT)
    try:
        df_wide.to_parquet(PARQUET_WIDE, index=False)
        df_long.to_parquet(PARQUET_LONG, index=False)
    except Exception as e:
        st.warning(f"Could not write Parquet ({e}). Using CSV in-memory this run.")

    return df_wide, df_long, months, CSV_DEFAULT.name

# ---- one-time load into session ----
try:
    df_wide, df_long, months, data_src_str = load_and_prepare_fast()
    st.session_state["df_wide"]   = df_wide
    st.session_state["df_long"]   = df_long
    st.session_state["months"]    = months
    st.session_state["data_src"]  = data_src_str
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    uploaded = st.file_uploader(
        "Upload dataset (CSV)", type=["csv"],
        help="Wide format: one row per community; monthly cols like Impact_2024-04, Tonnage_2024-04, ..."
    )
    st.caption("If no upload, the default CSV from the app folder is used.")
    show_popup_charts = st.toggle(
        "Show charts in popups (slower)", value=False,
        help="Renders mini charts inside each popup. Turn off for fast map updates."
    )

# ---------- Get Data from session ----------
df_wide  = st.session_state["df_wide"].copy()
df_long  = st.session_state["df_long"].copy()
months   = st.session_state["months"]
data_src = st.session_state["data_src"]

# ---------- Merge pincode centroids if needed ----------
PINCODE_LOOKUP = BASE_DIR / "pincode_centroids.csv"
if ("Lat" not in df_wide.columns or "Lon" not in df_wide.columns):
    if PINCODE_LOOKUP.exists():
        try:
            look = pd.read_csv(PINCODE_LOOKUP)
            look.columns = [c.strip() for c in look.columns]
            if {"Pincode","Lat","Lon"}.issubset(look.columns):
                look["Pincode"] = look["Pincode"].astype(str).str.strip()
                look["Lat"] = pd.to_numeric(look["Lat"], errors="coerce")
                look["Lon"] = pd.to_numeric(look["Lon"], errors="coerce")
                df_wide["Pincode"] = df_wide["Pincode"].astype(str).str.strip()
                df_wide = df_wide.merge(look[["Pincode","Lat","Lon"]], on="Pincode", how="left")
                st.caption(
                    "🗺️ Coordinates merged from `pincode_centroids.csv` "
                    f"(markers available for {(df_wide[['Lat','Lon']].notna().all(axis=1)).sum()} communities)."
                )
            else:
                st.warning("`pincode_centroids.csv` columns must be exactly: Pincode, Lat, Lon.")
        except Exception as e:
            st.warning(f"Could not read/merge `pincode_centroids.csv`: {e}")

# ---------- Title ----------
st.markdown("<h1>Smart Waste Analytics — FY 2024–25</h1>", unsafe_allow_html=True)
st.caption(f"Data source: {data_src}")

# ---------- Global Filters ----------
st.markdown("### 🔎 Global Filters")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.2])

city_opts = sorted(df_wide["City"].dropna().unique().tolist()) if "City" in df_wide else []
comm_opts = sorted(df_wide["Community"].dropna().unique().tolist()) if "Community" in df_wide else []
pin_opts  = sorted(df_wide["Pincode"].dropna().unique().tolist()) if "Pincode" in df_wide else []

with c1: sel_city = st.multiselect("City", city_opts, placeholder="All")
with c2: sel_comm = st.multiselect("Community", comm_opts, placeholder="All")
with c3: sel_pin  = st.multiselect("Pincode", pin_opts,  placeholder="All")
with c4:
    start_m, end_m = st.select_slider("Date range (month)", options=months, value=(months[0], months[-1]))

# Fast filter (avoid extra copies / casts)
def apply_filters(dfw, dfl):
    dfw_f = dfw
    if sel_city: dfw_f = dfw_f[dfw_f["City"].isin(sel_city)]
    if sel_comm: dfw_f = dfw_f[dfw_f["Community"].isin(sel_comm)]
    if sel_pin:  dfw_f = dfw_f[dfw_f["Pincode"].isin(sel_pin)]

    d0 = pd.to_datetime(start_m + "-01")
    d1 = pd.to_datetime(end_m + "-01")
    dfl_f = dfl[(dfl["Date"] >= d0) & (dfl["Date"] <= d1)]
    if sel_city: dfl_f = dfl_f[dfl_f["City"].isin([str(x) for x in sel_city])]
    if sel_comm: dfl_f = dfl_f[dfl_f["Community"].isin([str(x) for x in sel_comm])]
    if sel_pin:  dfl_f = dfl_f[dfl_f["Pincode"].isin([str(x) for x in sel_pin])]
    return dfw_f, dfl_f

dfw_filt, dfl_filt = apply_filters(df_wide, df_long)

# ---------- Summary KPIs ----------
@st.cache_data(show_spinner=False)
def kpi_row(dfl):
    def agg(metric, how="sum"):
        s = dfl.loc[dfl["Metric"] == metric, "Value"]
        if s.empty: return 0.0
        return float(s.sum() if how == "sum" else s.mean())
    return {
        "tonnage": agg("Tonnage", "sum"),
        "co2":     agg("CO2_Kgs_Averted", "sum"),
        "seg":     agg("Segregation_Compliance_Pct", "mean"),
        "hh":      agg("Households_Participating", "sum"),
    }

st.markdown("### 📊 Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
k1.metric("Communities", dfw_filt["Community"].nunique() if "Community" in dfw_filt else 0)
k2.metric("Cities", dfw_filt["City"].nunique() if "City" in dfw_filt else 0)

k = kpi_row(dfl_filt)
k3.metric("Total Tonnage", f"{k['tonnage']:,.0f}")
k4.metric("CO₂ Averted (kg)", f"{k['co2']:,.0f}")
k5.metric("Avg Segregation (%)", f"{k['seg']:.1f}")
k6.metric("Active Households", f"{k['hh']:,.0f}")
st.caption(f"Period: **{start_m} → {end_m}**")

# ---------- Tabs ----------
tab_map, tab_insights = st.tabs(["🗺️ 2D Map & Popups", "🧠 Insights"])

# ---------- Chart helpers ----------
def _brand_axes(fig, title=None):
    fig.update_layout(
        title=title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None),
        font=dict(family="Poppins", color=BRAND_PRIMARY, size=14),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(l=30, r=20, t=50, b=60),
    )
    fig.update_xaxes(
        title_font=dict(color=BRAND_PRIMARY, size=12),
        tickfont=dict(color=BRAND_PRIMARY, size=11),
        gridcolor="#EEE", zerolinecolor="#EEE",
    )
    fig.update_yaxes(
        title_font=dict(color=BRAND_PRIMARY, size=12),
        tickfont=dict(color=BRAND_PRIMARY, size=11),
        gridcolor="#EEE", zerolinecolor="#EEE",
    )
    return fig

# ---------- Cached data helpers ----------
@st.cache_data(show_spinner=False)
def monthly_series(dfl, community: str, metric: str):
    d = dfl[
        (dfl["Community"] == str(community)) &
        (dfl["Metric"] == metric)
    ][["Date", "Value"]].sort_values("Date").copy()
    return d

@st.cache_data(show_spinner=False)
def summarize_for_popup(dfl_filtered: pd.DataFrame, community_id: str, pincode: str|None):
    """
    Returns all popup KPIs based ONLY on current date-filtered data.
    Keys returned: tonnage, co2, households, seg_pct, trees
    """
    d = dfl_filtered
    d = d[d["Community"] == str(community_id)]
    if pincode is not None and "Pincode" in d.columns:
        d = d[d["Pincode"] == str(pincode)]

    def agg(metric: str, how="sum") -> float:
        s = d.loc[d["Metric"] == metric, "Value"]
        if s.empty: return 0.0
        return float(s.sum() if how == "sum" else s.mean())

    # Prefer dry waste if present
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_kg = 0.0
    for m in dry_candidates:
        s = d.loc[d["Metric"] == m, "Value"]
        if not s.empty:
            dry_kg = float(s.sum()); break

    co2   = dry_kg * CO2_PER_KG_DRY
    trees = dry_kg / KG_PER_TREE

    return {
        "tonnage":    agg("Tonnage", "sum"),
        "co2":        co2,
        "households": agg("Households_Participating", "sum"),
        "seg_pct":    agg("Segregation_Compliance_Pct", "mean"),
        "trees":      trees,
    }

# Mini chart images for POPUPS (cached)
def _to_data_uri(fig, w=340):
    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=180)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' style='width:{w}px;height:auto;border:0;'/>"

@st.cache_data(show_spinner=False)
def popup_charts_for_comm(dfl_filtered: pd.DataFrame, community_id: str):
    BRAND = BRAND_PRIMARY
    dm = dfl_filtered[dfl_filtered["Community"] == str(community_id)]
    if dm.empty:
        return "", ""

    dm = dm.copy()
    dm["MonthKey"] = dm["Date"].dt.to_period("M")

    # TONNAGE trend (as light line)
    bar_img = ""
    d_ton = dm[dm["Metric"] == "Tonnage"][["MonthKey", "Value"]]
    if not d_ton.empty:
        d_ton = d_ton.groupby("MonthKey", as_index=False)["Value"].sum().sort_values("MonthKey")
        xlab = [p.to_timestamp().strftime("%b") for p in d_ton["MonthKey"]]
        fig, ax = plt.subplots(figsize=(3.1, 1.6), dpi=120)
        ax.plot(xlab, d_ton["Value"], marker="o", lw=1.6, color=BRAND)
        ax.set_title("Tonnage", fontsize=9, color=BRAND, pad=2)
        ax.tick_params(axis="x", labelsize=8, colors=BRAND)
        ax.tick_params(axis="y", labelsize=8, colors=BRAND)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
        for s in ax.spines.values(): s.set_visible(False)
        ax.grid(alpha=0.12, axis="y")
        bar_img = _to_data_uri(fig, w=300)

    # CO2 donut
    donut_img = ""
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_month = None
    for m in dry_candidates:
        cur = dm[dm["Metric"] == m][["MonthKey", "Value"]]
        if not cur.empty:
            dry_month = cur; break
    if dry_month is not None:
        d = dry_month.groupby("MonthKey", as_index=False)["Value"].sum().sort_values("MonthKey")
        vals = (d["Value"] * CO2_PER_KG_DRY).clip(lower=0.0).to_numpy()
        labels = [p.to_timestamp().strftime("%b") for p in d["MonthKey"]]
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=120)
        wedges, _ = ax.pie(vals, wedgeprops=dict(width=0.45), startangle=90, colors=colors)
        ax.set(aspect="equal")
        ax.text(0, 0, "CO₂\nAverted", ha="center", va="center",
                fontsize=9, color=BRAND, fontweight="bold", linespacing=1.1)
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, frameon=False)
        donut_img = _to_data_uri(fig, w=220)

    return bar_img, donut_img

# ---------- Map Helpers ----------
def jitter_duplicates(df, lat_col="Lat", lon_col="Lon", jitter_deg=0.00025):
    """Spread markers that share the same (Lat, Lon) so each popup/tooltip works."""
    df = df.copy()
    gb = df.groupby([df[lat_col].round(6), df[lon_col].round(6)])
    for _, idx in gb.groups.items():
        n = len(idx)
        if n > 1:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = jitter_deg
            df.loc[idx, lat_col] = df.loc[idx, lat_col].to_numpy() + r * np.sin(angles)
            df.loc[idx, lon_col] = df.loc[idx, lon_col].to_numpy() + r * np.cos(angles)
    return df

# ========================= Map Tab =========================
with tab_map:
    has_latlon = (
        "Lat" in dfw_filt.columns and "Lon" in dfw_filt.columns and
        dfw_filt[["Lat","Lon"]].notna().all(axis=1).any()
    )

    if not has_latlon:
        st.warning("Map needs coordinates. Add **Lat/Lon** columns or merge a `pincode_centroids.csv`.")
        st.info("Click markers to see details here (after coordinates are available).")
        selected_comm, selected_pin = None, None
    else:
        valid = jitter_duplicates(dfw_filt.dropna(subset=["Lat", "Lon"]))
        lat0 = float(valid["Lat"].mean()); lon0 = float(valid["Lon"].mean())

        fmap = folium.Map(location=[lat0, lon0], zoom_start=11, tiles="cartodbpositron")
        cluster = MarkerCluster().add_to(fmap)

        # Prepare arrays (fast iteration)
        comm_arr = valid["Community"].astype(str).to_numpy()
        pin_arr  = valid["Pincode"].astype(str).to_numpy()
        lat_arr  = valid["Lat"].astype(float).to_numpy()
        lon_arr  = valid["Lon"].astype(float).to_numpy()
        city_arr = valid["City"].astype(str).to_numpy() if "City" in valid else np.array([""]*len(valid))

        # Date window (already used when making dfl_filt)
        for comm, pin, lat, lon, city in zip(comm_arr, pin_arr, lat_arr, lon_arr, city_arr):
            stats = summarize_for_popup(dfl_filt, community_id=comm, pincode=pin)

            # Build popup (charts optional)
            bar_img = donut_img = ""
            if show_popup_charts:
                try:
                    bar_img, donut_img = popup_charts_for_comm(dfl_filt, comm)
                except Exception:
                    bar_img = donut_img = ""

            popup_html = f"""
            <div style='font-family:Poppins; width:360px;'>
                <h4 style='margin:0 0 4px 0; color:#36204D;'>{comm}</h4>
                <div style='font-size:12px; color:#333;'>City: {city} | Pincode: {pin}</div>
                <hr style='margin:6px 0;'>

                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{TREE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['trees']:,.0f} Trees Saved</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{HOUSE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['households']:,.0f} Households Participating</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{RECYCLE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['seg_pct']:.1f}% Segregation</b></span>
                </div>

                {"<hr style='margin:8px 0;'><div style='margin-bottom:8px;'><b>CO₂ Averted</b>" + donut_img + "</div><div style='margin-top:6px;'><b>Tonnage</b>" + bar_img + "</div>" if show_popup_charts else ""}
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=BRAND_PRIMARY,
                fill=True,
                fill_color=BRAND_PRIMARY,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(f"{comm} • {pin}"),
                popup=folium.Popup(popup_html, max_width=420),
            ).add_to(cluster)

        st.markdown("##### Map")
        map_event = st_folium(
            fmap,
            height=ST_MAP_HEIGHT,
            use_container_width=True,
            returned_objects=[],              # keep bridge minimal
            key="main_leaflet_map_v1",        # stable so Streamlit doesn't remount
        )

        # Selection from click
        selected_comm, selected_pin = None, None
        if map_event and map_event.get("last_object_clicked_tooltip"):
            tip = map_event["last_object_clicked_tooltip"]  # "COMMUNITY • PINCODE"
            parts = [p.strip() for p in tip.split("•")]
            if len(parts) == 2:
                selected_comm, selected_pin = parts[0], parts[1]
            else:
                selected_comm = parts[0]
        elif not dfw_filt.empty:
            selected_comm = str(dfw_filt.iloc[0]["Community"])
            selected_pin  = str(dfw_filt.iloc[0]["Pincode"])

    # ---- Selection & Trends BELOW the map ----
    st.markdown("#### Trends (Selected Community)")
    if selected_comm is None:
        st.info("Click a marker to see selection details and trends here.")
    else:
        cA, cB, cC, cD = st.columns(4)
        stats_sel = summarize_for_popup(dfl_filt, community_id=selected_comm, pincode=selected_pin)
        cA.metric("Tonnage (kg)", f"{stats_sel['tonnage']:,.0f}")
        cB.metric("CO₂ Averted (kg)", f"{stats_sel['co2']:,.0f}")
        cC.metric("Households", f"{stats_sel['households']:,.0f}")
        cD.metric("Segregation % (avg)", f"{stats_sel['seg_pct']:.1f}")

        ton = monthly_series(dfl_filt, selected_comm, "Tonnage")
        if not ton.empty:
            fig_ton = px.line(
                ton, x="Date", y="Value",
                title="Tonnage over Time",
                labels={"Value": "Tonnage (kg)", "Date": "Date"},
                markers=True,
            )
            fig_ton.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                                  marker=dict(color=BRAND_PRIMARY))
            st.plotly_chart(_brand_axes(fig_ton), use_container_width=True)
        else:
            st.info("No tonnage data in this date range for the selected community.")

        dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
        dry = None
        for m in dry_candidates:
            s = monthly_series(dfl_filt, selected_comm, m)
            if not s.empty: dry = s; break

        if dry is not None and not dry.empty:
            co2 = dry.copy()
            co2["CO2_kg"] = (co2["Value"] * CO2_PER_KG_DRY).clip(lower=0.0)
            fig_co2 = px.line(
                co2, x="Date", y="CO2_kg",
                markers=True,
                title="CO₂ Averted (Calculated) over Time",
                labels={"CO2_kg": "CO₂ Averted (kg)", "Date": "Date"},
            )
            fig_co2.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                                  marker=dict(color=BRAND_PRIMARY, size=6))
            st.plotly_chart(_brand_axes(fig_co2), use_container_width=True)
        else:
            st.info("No dry/tonnage series available to compute CO₂ for this community.")

        seg = monthly_series(dfl_filt, selected_comm, "Segregation_Compliance_Pct")
        if not seg.empty:
            fig_seg = px.line(
                seg, x="Date", y="Value",
                markers=True,
                title="Segregation % over Time",
                labels={"Value": "Segregation (%)", "Date": "Date"},
            )
            fig_seg.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                                  marker=dict(color=BRAND_PRIMARY, size=6))
            st.plotly_chart(_brand_axes(fig_seg), use_container_width=True)

# ========================= Insights Tab =========================
with tab_insights:
    st.markdown("### 🧠 Auto Insights (All Cities in Selected Date Range)")

    d0 = pd.to_datetime(start_m + "-01")
    d1 = pd.to_datetime(end_m + "-01")
    dfl_date = df_long[(df_long["Date"] >= d0) & (df_long["Date"] <= d1)].copy()

    for col in ["City", "Community", "Pincode"]:
        if col in dfl_date.columns:
            dfl_date[col] = dfl_date[col].astype(str)

    sum_metrics  = ["Tonnage", "CO2_Kgs_Averted", "Households_Participating"]
    mean_metrics = ["Segregation_Compliance_Pct", "Impact"]

    city_sum = (
        dfl_date[dfl_date["Metric"].isin(sum_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value",
                     aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    city_mean = (
        dfl_date[dfl_date["Metric"].isin(mean_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value",
                     aggfunc="mean", fill_value=0.0)
        .reset_index()
    )

    colA, colB, colC, colD = st.columns(4)
    def _top_city(df, metric, how="max"):
        if df.empty or metric not in df.columns: return "—", 0.0
        row = df.loc[df[metric].idxmax()] if how == "max" else df.loc[df[metric].idxmin()]
        return str(row["City"]), float(row[metric])

    t_city, t_val = _top_city(city_sum, "Tonnage", "max")
    c_city, c_val = _top_city(city_sum, "CO2_Kgs_Averted", "max")
    h_city, h_val = _top_city(city_sum, "Households_Participating", "max")
    s_city, s_val = _top_city(city_mean, "Segregation_Compliance_Pct", "max")

    with colA: st.caption("Top city by Tonnage"); st.subheader(t_city); st.success(f"↑ {t_val:,.0f}")
    with colB: st.caption("Top city by CO₂ averted (kg)"); st.subheader(c_city); st.success(f"↑ {c_val:,.0f}")
    with colC: st.caption("Top city by Households"); st.subheader(h_city); st.success(f"↑ {h_val:,.0f}")
    with colD: st.caption("Highest Avg Segregation (%)"); st.subheader(s_city); st.success(f"↑ {s_val:,.1f}%")

    st.markdown("---")

    if not city_sum.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                city_sum.sort_values("Tonnage", ascending=False),
                x="City", y="Tonnage", text="Tonnage", title="Total Tonnage by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c2:
            fig = px.bar(
                city_sum.sort_values("CO2_Kgs_Averted", ascending=False),
                x="City", y="CO2_Kgs_Averted", text="CO2_Kgs_Averted", title="CO₂ Averted (kg) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                city_sum.sort_values("Households_Participating", ascending=False),
                x="City", y="Households_Participating", text="Households_Participating", title="Households by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c4:
            fig = px.bar(
                city_mean.sort_values("Segregation_Compliance_Pct", ascending=False),
                x="City", y="Segregation_Compliance_Pct", text="Segregation_Compliance_Pct",
                title="Avg Segregation (%) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

    st.markdown("---")

    # Top communities by Tonnage (across all cities)
    topN = 10
    comm_tonn = (
        dfl_date[dfl_date["Metric"] == "Tonnage"]
        .groupby(["City", "Community", "Pincode"], as_index=False)["Value"].sum()
        .rename(columns={"Value": "Tonnage"})
        .sort_values("Tonnage", ascending=False)
        .head(topN)
    )

    st.markdown(f"#### Top {topN} Communities by Tonnage (All Cities)")
    if not comm_tonn.empty:
        fig_comm = px.bar(
            comm_tonn, x="Community", y="Tonnage", color="City",
            title="Top Communities by Total Tonnage",
            labels={"Value": "Tonnage", "Community": "Community"},
        )
        fig_comm.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(_brand_axes(fig_comm), use_container_width=True)
        st.caption("Tip: Hover a bar to see its city and pincode.")
    else:
        st.info("No community tonnage available in this date range.")

    st.markdown("---")
    st.write("The table below reflects the **current filters** (city/community/pincode + date range).")
    filtered_csv = dfl_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Trends (filtered CSV)",
        data=filtered_csv,
        file_name="trends_filtered.csv",
        mime="text/csv",
        key="dl_trends_bottom",
    )
    st.dataframe(dfl_filt, use_container_width=True, height=420)


