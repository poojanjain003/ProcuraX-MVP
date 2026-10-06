import re
import html
from urllib.parse import quote
import base64
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# XGBoost on macOS may require the OpenMP runtime (libomp). Load it before
# importing xgboost so Apple Silicon installations can resolve the native
# library even when Homebrew's libomp is not on the default loader path.
def _load_macos_openmp_runtime():
    if sys.platform != "darwin":
        return
    candidates = [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ]
    try:
        import subprocess
        brew = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, timeout=3)
        if brew.returncode == 0:
            candidates.insert(0, str(Path(brew.stdout.strip()) / "lib" / "libomp.dylib"))
    except Exception:
        pass
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                return
        except Exception:
            continue

import sys
import ctypes
_load_macos_openmp_runtime()
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
import plotly.express as px
import plotly.graph_objects as go
import database

st.set_page_config(page_title="ProcuraX | AI Procurement Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# ProcuraX design system
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --navy:#0B2545; --navy2:#123B63; --teal:#13BFAE; --teal2:#0A9E92; --ink:#12233F; --muted:#64748B; --line:#DCE7EF; --bg:#F5F9FC; --card:#FFFFFF; --red:#D64545; --amber:#D59618; --green:#15966B; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#F8FBFD 0%,#FFFFFF 58%,#F2FBF9 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
.brand { padding: 10px 8px 22px; border-bottom:1px solid var(--line); margin-bottom:16px; }
.brand-name { font-size:2rem; font-weight:800; letter-spacing:-1.4px; color:var(--navy); line-height:1; }
.brand-name span { color:var(--teal); }
.brand-sub { font-size:.67rem; letter-spacing:1.7px; color:#57708B; margin-top:5px; font-weight:700; }
.nav-label { font-size:.67rem; letter-spacing:1.5px; color:#94A3B8; font-weight:700; margin:17px 0 6px 8px; }
[data-testid="stSidebar"] .stRadio label { padding:7px 8px; border-radius:10px; font-weight:600; color:#263A53; }
[data-testid="stSidebar"] .stRadio label:hover { background:#EFF9F8; }
.status { background:#ECFDF7; border:1px solid #B9EDE1; color:#087F69; border-radius:10px; padding:10px 11px; font-size:.78rem; font-weight:600; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:5px 0 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
.eyebrow { color:var(--teal2); font-size:.75rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; }
.page-title { font-size:2rem; font-weight:800; color:var(--navy); letter-spacing:-1px; margin:2px 0 2px; }
.page-sub { color:var(--muted); font-size:.96rem; }
.hero { background:linear-gradient(110deg,#0B2545 0%,#123B63 58%,#0A5960 100%); border-radius:18px; padding:25px 28px; color:#fff; position:relative; overflow:hidden; margin-bottom:18px; }
.hero:after { content:'X'; position:absolute; right:35px; top:-36px; font-size:220px; font-weight:800; color:rgba(19,191,174,.13); transform:skewX(-10deg); }
.hero h1 { font-size:2rem; margin:0; letter-spacing:-1px; position:relative; z-index:1; }
.hero h1 span { color:#48D8C9; }
.hero p { margin:8px 0 0; color:#D7E7F4; position:relative; z-index:1; }
.pill { display:inline-block; border-radius:999px; padding:5px 9px; font-size:.67rem; font-weight:800; letter-spacing:.4px; }
.pill-red { background:#FEE2E2; color:#B91C1C; } .pill-amber{background:#FFEDD5;color:#C2410C;} .pill-green{background:#DCFCE7;color:#15803D;} .pill-teal{background:#DDF8F4;color:#087F76;}
.metric { background:var(--card); border:1px solid var(--line); border-radius:15px; padding:17px 18px; min-height:112px; box-shadow:0 2px 8px rgba(11,37,69,.03); }
.health-main-card { position:relative; }
.health-info { position:absolute; top:14px; right:14px; width:22px; height:22px; border:1px solid #B9C9D8; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#64748B; font-size:.72rem; font-weight:800; cursor:help; background:#FFFFFF; }
.health-info .health-tooltip { visibility:hidden; opacity:0; position:absolute; z-index:20; top:30px; right:0; width:275px; padding:11px 13px; background:#FFFFFF; border:1px solid #D7E2EC; border-radius:10px; box-shadow:0 8px 24px rgba(11,37,69,.12); color:#31566A; font-size:.76rem; line-height:1.45; font-weight:500; text-align:left; pointer-events:none; transition:opacity .12s ease; }
.health-info:hover .health-tooltip { visibility:visible; opacity:1; }
.metric .label { color:#64748B; font-size:.80rem; font-weight:600; } .metric .value { color:var(--navy); font-size:1.8rem; font-weight:800; margin-top:6px; } .metric .delta { color:#15966B; font-size:.79rem; margin-top:4px; }
.section-title { font-size:1.12rem; font-weight:800; color:var(--navy); margin:18px 0 9px; }
.section-sub { color:#64748B; font-size:.86rem; margin-top:-6px; margin-bottom:10px; }
.ai-card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:14px; padding:15px 17px; margin-bottom:10px; box-shadow:0 2px 8px rgba(11,37,69,.03); }
.ai-card.critical { border-left-color:var(--red); } .ai-card.watch { border-left-color:var(--amber); }
.ai-title { color:var(--navy); font-size:1.06rem; font-weight:800; margin:5px 0 7px; } .ai-meta { color:#52657B; font-size:.86rem; } .ai-reason { color:#3D526A; font-size:.88rem; margin-top:8px; }
[class*="st-key-ai_decision_card"] { background:#FFFFFF !important; }
.insight { background:#F0FBF9; border:1px solid #C7EEE7; border-radius:13px; padding:13px 15px; }
.workflow-status-card { min-height:78px; box-sizing:border-box; margin-bottom:0 !important; }
.workflow-task { background:#ECFDF3; border:1px solid #BFE8CE; border-radius:13px; padding:9px 13px; margin:0 0 0 0; color:#1E7A3D; line-height:1.35; min-height:56px; box-sizing:border-box; }
.workflow-rejection-box { background:#FFF7F7; border:1px solid #FECACA; border-radius:13px; padding:12px 14px; margin:16px 0 0; box-sizing:border-box; }
.final-decision-card { margin-bottom:16px !important; }
.final-decision-action-gap { height:16px; }
.purchase-task-spacer { height:16px; }
.workflow-message-spacer { height:16px; }
.workflow-message { margin:0; }
.purchase-task-row { display:flex; align-items:center; gap:10px; margin:12px 0 2px; }
.purchase-task-copy { flex:1; }
.callback-col { padding-top:7px; }
.callback-col button { background:#FEE2E2 !important; color:#B91C1C !important; border:1px solid #FCA5A5 !important; font-weight:700 !important; }
.callback-col button:hover { background:#FECACA !important; border-color:#F87171 !important; }
.workflow-task b:first-child { display:block; font-size:.76rem; letter-spacing:.05em; margin-bottom:3px; }
.insight-title { color:#087F76; font-weight:800; font-size:.82rem; } .insight p { color:#31566A; font-size:.86rem; margin:5px 0 0; }
.panel { background:#fff; border:1px solid var(--line); border-radius:15px; padding:17px; box-shadow:0 2px 8px rgba(11,37,69,.03); }
.command-card-shell { position:relative; }
.command-review-icon { position:absolute; top:13px; right:13px; width:30px; height:30px; display:flex; align-items:center; justify-content:center; background:#15966B; color:#FFFFFF !important; border-radius:8px; text-decoration:none !important; font-size:15px; font-weight:800; line-height:1; box-shadow:0 2px 5px rgba(21,150,107,.18); }
.command-review-icon:hover { background:#0A9E92; color:#FFFFFF !important; }
.command-review-btn { position:absolute; top:13px; right:13px; display:inline-flex; align-items:center; justify-content:center; min-width:78px; height:32px; padding:0 14px; background:#15966B; color:#FFFFFF !important; border-radius:8px; text-decoration:none !important; font-size:.78rem; font-weight:750; line-height:1; box-shadow:0 2px 5px rgba(21,150,107,.18); }
.command-review-btn:hover { background:#0A9E92; color:#FFFFFF !important; }

.kpi-label { color:#64748B; font-size:.72rem; } .kpi-value { color:var(--navy); font-size:1.35rem; font-weight:800; }
hr { border-color:var(--line) !important; }
[data-testid="stMetricValue"] { color:var(--navy); }
.stButton > button { border-radius:9px; font-weight:700; border:1px solid #BFD1DF; }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); color:white; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
/* v99: white select/dropdown controls to match the ProcuraX card aesthetic */
[data-baseweb="select"] > div { background:#FFFFFF !important; }
[data-baseweb="select"] input { background:#FFFFFF !important; }
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"] { background:#FFFFFF !important; }
/* v100: keep the Annual purchase volume control consistent with white inputs */
[class*="st-key-neg_volume"] [data-testid="stNumberInput"] > div,
[class*="st-key-neg_volume"] [data-testid="stNumberInput"] [data-baseweb="input"] > div,
[class*="st-key-neg_volume"] [data-testid="stNumberInput"] input { background:#FFFFFF !important; }
.footer { margin-top:30px; padding-top:12px; border-top:1px solid var(--line); color:#8091A4; font-size:.7rem; text-align:center; }

/* v120: Master Admin console */
.admin-summary-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:4px 0 18px; }
.admin-stat { background:#FFFFFF; border:1px solid var(--line); border-radius:14px; padding:14px 16px; box-shadow:0 2px 8px rgba(11,37,69,.03); }
.admin-stat-label { color:#64748B; font-size:.74rem; font-weight:700; }
.admin-stat-value { color:var(--navy); font-size:1.45rem; font-weight:800; margin-top:4px; }
.admin-stat-sub { color:#8091A4; font-size:.72rem; margin-top:2px; }
.admin-section { background:#FFFFFF; border:1px solid var(--line); border-radius:15px; padding:18px 20px; margin-bottom:16px; box-shadow:0 2px 8px rgba(11,37,69,.03); }
.admin-section-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:14px; }
.admin-section-title { color:var(--navy); font-size:1.04rem; font-weight:800; }
.admin-section-sub { color:#64748B; font-size:.78rem; margin-top:3px; }
.admin-table { width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid var(--line); border-radius:12px; font-size:.78rem; }
.admin-table th { background:#F7FAFC; color:#64748B; font-size:.68rem; letter-spacing:.55px; text-transform:uppercase; font-weight:800; padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }
.admin-table td { padding:10px 12px; color:#263A53; border-bottom:1px solid #EDF2F6; vertical-align:middle; }
.admin-table tr:last-child td { border-bottom:none; }
.admin-user-name { color:var(--navy); font-weight:750; }
.admin-username { color:#8091A4; font-size:.70rem; margin-top:2px; }
.admin-status { display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:4px 8px; font-size:.65rem; font-weight:800; }
.admin-status.active { background:#DCFCE7; color:#15803D; }
.admin-status.disabled { background:#F1F5F9; color:#64748B; }
.admin-role { color:#087F76; font-weight:700; }
.admin-note { background:#F0FBF9; border:1px solid #C7EEE7; border-radius:11px; padding:10px 12px; color:#31566A; font-size:.74rem; line-height:1.45; margin-top:12px; }
.admin-danger-note { background:#FFF8F8; border:1px solid #FECACA; border-radius:11px; padding:10px 12px; color:#7F1D1D; font-size:.74rem; line-height:1.45; margin-top:12px; }
@media (max-width: 900px) { .admin-summary-grid { grid-template-columns:1fr; } }

.nav-group-title { font-size:.66rem; letter-spacing:1.6px; color:#7890A7; font-weight:800; margin:18px 0 6px 8px; }
[data-testid="stSidebar"] .nav-item, [data-testid="stSidebar"] .nav-active { margin:0 0 3px; }
[data-testid="stSidebar"] .nav-item button, [data-testid="stSidebar"] .nav-active button {
    border:1px solid transparent !important; border-radius:9px !important; text-align:left !important;
    padding:9px 12px !important; font-weight:650 !important; box-shadow:none !important;
}
[data-testid="stSidebar"] .nav-item button { background:transparent !important; color:#263A53 !important; }
[data-testid="stSidebar"] .nav-item button:hover { background:#EAF8F6 !important; color:#087F76 !important; }
[data-testid="stSidebar"] .nav-active button {
    background:linear-gradient(90deg,#0B2545,#123B63) !important; color:#FFFFFF !important;
    box-shadow:0 4px 12px rgba(11,37,69,.14) !important;
}
[data-testid="stSidebar"] .nav-active button:hover { background:linear-gradient(90deg,#0B2545,#123B63) !important; color:#FFFFFF !important; }


/* Compact navigation: all grouped tools visible in one normal MacBook viewport */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}
[data-testid="stSidebar"] .brand {
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] .nav-group-title {
    margin: 8px 0 3px 8px !important;
    line-height: 1.1 !important;
}
[data-testid="stSidebar"] .nav-item,
[data-testid="stSidebar"] .nav-active {
    margin: 0 0 1px 0 !important;
}
[data-testid="stSidebar"] .nav-item button,
[data-testid="stSidebar"] .nav-active button {
    min-height: 36px !important;
    height: 36px !important;
    padding: 5px 11px !important;
    margin: 0 !important;
    font-size: 0.88rem !important;
    line-height: 1.05 !important;
}


/* ULTRA-COMPACT SIDEBAR NAVIGATION */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.15rem !important;
    padding-bottom: 0.15rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}
[data-testid="stSidebar"] .brand {
    margin-top: 0 !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .nav-group-title {
    margin: 5px 0 1px 8px !important;
    padding: 0 !important;
    line-height: 1 !important;
    height: auto !important;
}
[data-testid="stSidebar"] .nav-item,
[data-testid="stSidebar"] .nav-active {
    margin: 0 !important;
    padding: 0 !important;
    height: 32px !important;
    line-height: 32px !important;
}
[data-testid="stSidebar"] .nav-item button,
[data-testid="stSidebar"] .nav-active button {
    min-height: 32px !important;
    height: 32px !important;
    padding: 3px 10px !important;
    margin: 0 !important;
    font-size: 0.82rem !important;
    line-height: 1 !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] hr {
    margin: 5px 0 !important;
}


/* v17 sidebar layout */
[data-testid="stSidebar"] > div:first-child { padding-top: .15rem !important; }
[data-testid="stSidebar"] .brand {
    padding: 5px 8px 9px !important;
    margin: 0 0 5px !important;
}
[data-testid="stSidebar"] .brand-name { font-size: 1.8rem !important; }
[data-testid="stSidebar"] .brand-sub { margin-top: 3px !important; }
[data-testid="stSidebar"] .nav-group-title {
    margin: 4px 0 1px 8px !important;
}
[data-testid="stSidebar"] .nav-item,
[data-testid="stSidebar"] .nav-active {
    margin: 0 !important;
    height: 31px !important;
}
[data-testid="stSidebar"] .nav-item button,
[data-testid="stSidebar"] .nav-active button {
    min-height: 31px !important;
    height: 31px !important;
    padding: 2px 10px !important;
}

/* v38: polished sidebar hierarchy and bottom utility separation */
[data-testid="stSidebar"] .workflow-heading {
    margin: 6px 0 3px 8px !important;
    padding: 0 !important;
    color: #183452 !important;
    font-size: .70rem !important;
    font-weight: 750 !important;
    letter-spacing: .11em !important;
    text-transform: uppercase !important;
    line-height: 1.1 !important;
}
[data-testid="stSidebar"] .sidebar-bottom {
    margin-top: 8px !important;
    padding-top: 7px !important;
    border-top: 1px solid #DCE7EF !important;
}
[data-testid="stSidebar"] .sidebar-bottom .status {
    padding: 7px 9px !important;
    font-size: .70rem !important;
    line-height: 1.25 !important;
    margin: 0 0 6px !important;
}
[data-testid="stSidebar"] .sidebar-upload-label {
    color:#52657B !important; font-size:.69rem !important; font-weight:650 !important; margin:0 0 3px 1px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { padding: 5px !important; min-height: 48px !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div { font-size: .66rem !important; }

/* v41: explicit sidebar section separation — widgets are Streamlit siblings, so spacing is applied to their real containers. */
[data-testid="stSidebar"] > div:first-child {
    padding-top: .10rem !important;
    padding-bottom: .35rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .03rem !important; }
[data-testid="stSidebar"] [data-testid="element-container"] { margin:0 !important; padding:0 !important; }

/* Account block */
[data-testid="stSidebar"] .brand { padding:4px 8px 7px !important; margin:0 0 16px !important; }
[data-testid="stSidebar"] .brand-name { font-size:1.70rem !important; }
[data-testid="stSidebar"] .brand-sub { font-size:.60rem !important; margin-top:2px !important; }
[data-testid="stSidebar"] .manager-card { margin-bottom:3px !important; }

/* Sign-out gets its own breathing room before the workflow heading. */
[data-testid="stSidebar"] .signout-wrap { margin:0 !important; padding:0 !important; }
[data-testid="stSidebar"] .signout-wrap + .workflow-heading { margin-top:0 !important; }
[data-testid="stSidebar"] .workflow-heading {
    margin:9px 0 4px 8px !important;
    padding:0 !important;
    color:#183452 !important;
    font-size:.70rem !important;
    font-weight:750 !important;
    letter-spacing:.11em !important;
    text-transform:uppercase !important;
    line-height:1.1 !important;
}

/* Core workflow: tight, consistent button rhythm. */
[data-testid="stSidebar"] .nav-item,
[data-testid="stSidebar"] .nav-active {
    margin:0 0 1px 0 !important;
    padding:0 !important;
    height:27px !important;
}
[data-testid="stSidebar"] .nav-item button,
[data-testid="stSidebar"] .nav-active button {
    min-height:27px !important;
    height:27px !important;
    padding:1px 9px !important;
    margin:0 !important;
    font-size:.79rem !important;
    line-height:1 !important;
    border-radius:7px !important;
}
[data-testid="stSidebar"] button[kind="secondary"] {
    min-height:27px !important;
    height:27px !important;
    padding:1px 9px !important;
    font-size:.79rem !important;
}

/* IMPORTANT: sidebar-bottom used to be an HTML wrapper around Streamlit widgets.
   Streamlit renders those widgets as sibling DOM nodes, so the old wrapper could not
   reliably create separation. These selectors target the actual rendered siblings. */
[data-testid="stSidebar"] .sidebar-bottom-anchor {
    height:10px !important;
    margin:7px 0 3px !important;
    border-top:1px solid #DCE7EF !important;
}
[data-testid="stSidebar"] .sidebar-status-card {
    box-sizing:border-box !important;
    min-height:50px !important;
    padding:7px 9px !important;
    margin:0 0 16px !important;
    border:1px solid #B9EDE1 !important;
    border-radius:9px !important;
    background:#ECFDF7 !important;
    color:#087F69 !important;
    font-size:.69rem !important;
    line-height:1.25 !important;
}
[data-testid="stSidebar"] .sidebar-data-label {
    display:block !important;
    color:#52657B !important;
    font-size:.69rem !important;
    font-weight:700 !important;
    margin:0 0 4px 1px !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone {
    margin:0 !important;
    padding:0 !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone [data-testid="stFileUploaderDropzone"] {
    box-sizing:border-box !important;
    min-height:54px !important;
    height:54px !important;
    padding:5px 7px !important;
    border-radius:9px !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone [data-testid="stFileUploaderDropzoneInstructions"] {
    gap:0 !important;
    margin:0 !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone [data-testid="stFileUploaderDropzoneInstructions"] > div {
    font-size:.63rem !important;
    line-height:1.1 !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone [data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size:.58rem !important;
}
[data-testid="stSidebar"] .sidebar-upload-zone button {
    min-height:27px !important;
    height:27px !important;
    padding:1px 8px !important;
    font-size:.70rem !important;
}

/* v58: final sidebar hierarchy and spacing. */
/* Keep ProcuraX branding at the very top, with no separator underneath it. */
[data-testid="stSidebar"] .brand {
    border-bottom:none !important;
    margin-top:-58px !important;
    margin-bottom:16px !important;
    padding-bottom:7px !important;
}
/* Account section follows the branding cleanly; no line is placed above it. */
[data-testid="stSidebar"] .manager-card {
    margin-top:0 !important;
    margin-bottom:8px !important;
    min-height:112px !important;
    box-sizing:border-box !important;
}
/* The separator occupies the actual gap between Sign out and Procurement Operations,
   with the line vertically centered in that gap. */
[data-testid="stSidebar"] .account-separator {
    height:34px !important;
    min-height:34px !important;
    width:100% !important;
    margin:0 !important;
    padding:0 !important;
    position:relative !important;
    box-sizing:border-box !important;
}
[data-testid="stSidebar"] .account-separator::after {
    content:"" !important;
    position:absolute !important;
    left:8px !important;
    right:8px !important;
    top:17px !important;
    border-top:1px solid #DCE7EF !important;
}
[data-testid="stSidebar"] .workflow-heading {
    margin-top:10px !important;
    margin-bottom:10px !important;
}
[data-testid="stSidebar"] .sidebar-bottom-anchor {
    height:18px !important;
    min-height:18px !important;
    margin:10px 0 5px !important;
    border-top:1px solid #DCE7EF !important;
}
[data-testid="stSidebar"] .utility-gap {
    height:0 !important;
    min-height:0 !important;
    width:100% !important;
    margin:0 !important;
}
[data-testid="stSidebar"] .upload-separator {
    height:18px !important;
    min-height:18px !important;
    width:100% !important;
    margin:0 0 6px !important;
    padding:0 !important;
    position:relative !important;
    box-sizing:border-box !important;
}
[data-testid="stSidebar"] .upload-separator::after {
    content:"" !important;
    position:absolute !important;
    left:8px !important;
    right:8px !important;
    top:8px !important;
    border-top:1px solid #DCE7EF !important;
}
/* The uploader stands alone; no label or text is placed behind it. */
[data-testid="stSidebar"] .sidebar-data-label {
    display:none !important;
}
/* File uploader is a real Streamlit sibling, so target it directly. */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    margin:0 !important;
    padding:0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    box-sizing:border-box !important;
    min-height:54px !important;
    height:54px !important;
    padding:5px 7px !important;
    border-radius:9px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
    gap:0 !important;
    margin:0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div {
    font-size:.63rem !important;
    line-height:1.1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size:.58rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    min-height:27px !important;
    height:27px !important;
    padding:1px 8px !important;
    font-size:.70rem !important;
}

/* v53: negotiation intelligence card sizing and expander breathing room */
.negotiation-metric {
    box-sizing:border-box !important;
    min-height:140px !important;
    height:140px !important;
    overflow:hidden !important;
}
.negotiation-metric .value {
    overflow-wrap:anywhere !important;
    word-break:normal !important;
}
/* Give expander content a real inset so its inner recommendation cards never touch
   or visually merge with the outer expander border. */
[data-testid="stExpander"] details > div {
    box-sizing:border-box !important;
    padding:22px 24px 24px !important;
}
[data-testid="stExpander"] details > div .panel,
[data-testid="stExpander"] details > div .insight {
    margin-bottom:10px !important;
}
[data-testid="stExpander"] details > div > div:last-child .panel,
[data-testid="stExpander"] details > div > div:last-child .insight {
    margin-bottom:0 !important;
}

/* ProcuraX login — refined v31: clean, centered, single-screen */
body:has(input[aria-label="Username"]) { overflow:hidden !important; }
body:has(input[aria-label="Username"]) header[data-testid="stHeader"],
body:has(input[aria-label="Username"]) footer,
body:has(input[aria-label="Username"]) [data-testid="stSidebar"] { display:none !important; }
body:has(input[aria-label="Username"]) .stApp,
body:has(input[aria-label="Username"]) [data-testid="stAppViewContainer"],
body:has(input[aria-label="Username"]) [data-testid="stAppViewContainer"] > .main {
    background:#FFFFFF !important;
}
body:has(input[aria-label="Username"]) [data-testid="stAppViewContainer"] > .main {
    min-height:100vh !important;
}
body:has(input[aria-label="Username"]) .block-container {
    max-width:none !important; width:100% !important; height:100vh !important; min-height:100vh !important;
    padding:0 !important; margin:0 !important; display:flex !important; align-items:center !important;
    justify-content:center !important; overflow:hidden !important; position:relative !important;
    background:
      radial-gradient(circle at 50% 50%, rgba(238,251,249,.32) 0, rgba(255,255,255,0) 42%),
      #FFFFFF !important;
}
/* Soft report-cover-inspired X accents — decoration only, never an image */
body:has(input[aria-label="Username"]) .block-container::before,
body:has(input[aria-label="Username"]) .block-container::after {
    content:""; position:fixed; pointer-events:none; z-index:0;
    width:310px; height:310px; opacity:.18;
    background:
      linear-gradient(45deg, transparent 39%, #13BFAE 40%, #13BFAE 60%, transparent 61%),
      linear-gradient(-45deg, transparent 39%, #13BFAE 40%, #13BFAE 60%, transparent 61%);
    filter:blur(.15px);
}
body:has(input[aria-label="Username"]) .block-container::before {
    left:-138px; top:-142px;
    transform:rotate(0deg);
}
body:has(input[aria-label="Username"]) .block-container::after {
    right:-138px; bottom:-142px;
    transform:rotate(0deg);
}
body:has(input[aria-label="Username"]) [data-testid="stVerticalBlock"] { z-index:1; }
body:has(input[aria-label="Username"]) [data-testid="stHorizontalBlock"] {
    width:100% !important; max-width:520px !important; margin:0 auto !important;
    align-items:stretch !important; gap:0 !important;
    transform:translateY(-10px);
}
body:has(input[aria-label="Username"]) [data-testid="stColumn"] { padding:0 !important; }
body:has(input[aria-label="Username"]) [data-testid="stColumn"]:first-child,
body:has(input[aria-label="Username"]) [data-testid="stColumn"]:last-child { display:none !important; }
body:has(input[aria-label="Username"]) [data-testid="stColumn"]:nth-child(2) {
    width:520px !important; max-width:520px !important; flex:1 1 520px !important;
}
.login-shell {
    width:100%; box-sizing:border-box; background:rgba(255,255,255,.97); padding:0 18px;
    position:relative; z-index:2;
}
.login-logo {
    font-size:3.0rem; font-weight:800; letter-spacing:-2.7px; color:#0B2545;
    line-height:1; margin:0 0 8px;
}
.login-logo span { color:#13BFAE; }
.login-sub {
    color:#57708B; font-size:.70rem; letter-spacing:2.5px; font-weight:800; margin:0 0 25px;
}
.login-line { width:62px; height:4px; background:#13BFAE; border-radius:3px; margin:0 0 20px; }
.login-title {
    font-size:2.25rem; font-weight:800; color:#0B2545; letter-spacing:-1.5px; margin:0 0 7px;
}
.login-desc { color:#64748B; font-size:.98rem; margin:0 0 20px; }
body:has(input[aria-label="Username"]) label {
    color:#0B2545 !important; font-size:.86rem !important; font-weight:600 !important;
}
body:has(input[aria-label="Username"]) input {
    height:50px !important; box-sizing:border-box !important; border:1px solid #D7E1E9 !important;
    border-radius:10px !important; background:#FFFFFF !important; color:#0B2545 !important;
    font-size:.98rem !important; padding-left:15px !important;
}
body:has(input[aria-label="Username"]) input:focus {
    border-color:#13BFAE !important; box-shadow:0 0 0 2px rgba(19,191,174,.10) !important;
}
body:has(input[aria-label="Username"]) [data-testid="stTextInput"] { margin-bottom:8px !important; }
body:has(input[aria-label="Username"]) [data-testid="stCheckbox"] { margin-top:0 !important; }
body:has(input[aria-label="Username"]) [data-testid="stCheckbox"] label {
    font-weight:500 !important; color:#334155 !important;
}
body:has(input[aria-label="Username"]) .stButton button,
body:has(input[aria-label="Username"]) button[kind="primary"] {
    height:50px !important; border-radius:10px !important; background:#13BFAE !important;
    color:#FFFFFF !important; border:1px solid #13BFAE !important; font-weight:700 !important; font-size:.98rem !important;
}
body:has(input[aria-label="Username"]) .stButton button:hover {
    background:#0FAF9F !important; border-color:#0FAF9F !important;
}
.login-forgot { text-align:right; color:#0FAF9F; font-size:.80rem; font-weight:700; padding-top:5px; }
.login-footer {
    margin-top:29px; padding-top:14px; border-top:1px solid #E4EBF1; color:#6E8197;
    font-size:.73rem; letter-spacing:2.1px; font-weight:700; text-align:left;
}
.login-footer span { color:#13BFAE; padding:0 9px; }
body:has(input[aria-label="Username"]) [data-testid="stForm"] {
    border:0 !important; padding:0 !important; background:transparent !important;
}
body:has(input[aria-label="Username"]) [data-testid="stForm"] > div { gap:.16rem !important; }
@media (max-height: 760px) {
    body:has(input[aria-label="Username"]) [data-testid="stHorizontalBlock"] { transform:translateY(-6px); }
    body:has(input[aria-label="Username"]) .login-sub { margin-bottom:18px; }
    body:has(input[aria-label="Username"]) .login-line { margin-bottom:15px; }
    body:has(input[aria-label="Username"]) .login-title { font-size:2.05rem; }
    body:has(input[aria-label="Username"]) .login-desc { margin-bottom:14px; }
    body:has(input[aria-label="Username"]) input,
    body:has(input[aria-label="Username"]) .stButton button { height:46px !important; }
    body:has(input[aria-label="Username"]) .login-footer { margin-top:20px; padding-top:11px; }
}
@media (max-width: 700px) {
    body:has(input[aria-label="Username"]) .block-container { align-items:center !important; }
    body:has(input[aria-label="Username"]) [data-testid="stHorizontalBlock"] { max-width:100% !important; transform:none; }
    body:has(input[aria-label="Username"]) [data-testid="stColumn"]:nth-child(2) { width:100% !important; max-width:100% !important; }
    .login-shell { padding:0 24px; }
}


/* v60: add deliberate breathing room between the section heading and first workflow tab.
   This keeps the Procurement Operations label visually distinct from Command Center
   without changing the established Sign out separator or overall sidebar hierarchy. */
[data-testid="stSidebar"] .workflow-heading {
    margin-top:10px !important;
    margin-bottom:10px !important;
}

/* v59: sidebar viewport and uploader polish.
   Keep the AI engine card fully visible on first view; place the uploader below the fold. */
[data-testid="stSidebar"] .sidebar-status-card {
    margin-bottom:0 !important;
}
[data-testid="stSidebar"] .upload-separator {
    height:46px !important;
    min-height:46px !important;
    margin:0 0 8px !important;
}
[data-testid="stSidebar"] .upload-separator::after {
    top:23px !important;
}
/* Let the uploader size itself around its text and Browse button.
   The old fixed 54px height caused the button to escape the card. */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    box-sizing:border-box !important;
    width:100% !important;
    min-height:92px !important;
    height:auto !important;
    padding:9px 9px 10px !important;
    border-radius:10px !important;
    overflow:visible !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
    width:100% !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:flex-start !important;
    gap:3px !important;
    margin:0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div {
    font-size:.68rem !important;
    line-height:1.2 !important;
    white-space:normal !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size:.61rem !important;
    line-height:1.2 !important;
    white-space:normal !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    position:static !important;
    display:block !important;
    width:auto !important;
    max-width:100% !important;
    min-height:29px !important;
    height:29px !important;
    margin:5px 0 0 !important;
    padding:2px 10px !important;
    font-size:.70rem !important;
    box-sizing:border-box !important;
}

/* v62: account card refinement. Keep the sign-out control visually inside the
   compact manager card instead of letting it spill below the green panel. */
[data-testid="stSidebar"] .manager-card {
    height:80px !important;
    min-height:80px !important;
    max-height:80px !important;
    box-sizing:border-box !important;
    overflow:visible !important;
    padding:10px 12px 40px !important;
    margin:0 !important;
    border-radius:12px !important;
}
/* The Sign out Streamlit widget immediately follows the manager-card markdown.
   Pull it upward so the button sits within the lower part of the green card. */
[data-testid="stSidebar"] .manager-card + [data-testid="stElementContainer"] {
    position:relative !important;
    z-index:5 !important;
    margin-top:-46px !important;
    margin-bottom:0 !important;
}
[data-testid="stSidebar"] .manager-card + [data-testid="stElementContainer"] button {
    min-height:40px !important;
    height:40px !important;
    border-radius:11px !important;
    background:#FFFFFF !important;
    box-sizing:border-box !important;
}
/* Preserve equal breathing room around the Procurement Operations heading. */
[data-testid="stSidebar"] .workflow-heading {
    margin-top:10px !important;
    margin-bottom:10px !important;
}


/* v65: compact combined account section. The green card contains the account
   details, while the Sign out control forms the lower part of the same visual card.
   This removes the large empty area without letting the button spill outside. */
[data-testid="stSidebar"] .manager-card {
    height:68px !important;
    min-height:68px !important;
    max-height:68px !important;
    box-sizing:border-box !important;
    overflow:hidden !important;
    padding:9px 12px 4px !important;
    margin:0 !important;
    border-radius:12px 12px 0 0 !important;
    border-bottom:none !important;
}
[data-testid="stSidebar"] .manager-card + [data-testid="stElementContainer"] {
    position:relative !important;
    z-index:5 !important;
    margin:0 !important;
    padding:0 !important;
    background:#F0FBF9 !important;
    border-left:1px solid #C7EEE7 !important;
    border-right:1px solid #C7EEE7 !important;
    border-bottom:1px solid #C7EEE7 !important;
    border-radius:0 0 12px 12px !important;
    box-sizing:border-box !important;
}
[data-testid="stSidebar"] .manager-card + [data-testid="stElementContainer"] button {
    min-height:38px !important;
    height:38px !important;
    width:100% !important;
    border-radius:10px !important;
    background:#FFFFFF !important;
    box-sizing:border-box !important;
    margin:0 !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# User authentication + procurement domains
# -----------------------------
DEFAULT_USERS = {
    "poojan": {"name": "Poojan Jain", "password": "Poojan@123", "category": "Refrigerator Category", "role": "manager", "job_role": "Appliance Procurement Manager", "purchaser": "Arjun Mehta"},
    "neel": {"name": "Neel Bhinde", "password": "Neel@123", "category": "Air-Conditioning Category", "role": "manager", "job_role": "Appliance Procurement Manager", "purchaser": "Riya Shah"},
    "yash": {"name": "Yash Kale", "password": "Yash@123", "category": "Washing Machine Category", "role": "manager", "job_role": "Appliance Procurement Manager", "purchaser": "Karan Patel"},
    "admin": {"name": "ProcuraX Master Admin", "password": "Admin@123", "category": "All Categories", "role": "admin", "job_role": "Master Administrator", "purchaser": "—"},
}

def _refresh_users_from_db():
    users = database.list_users()
    result = {}
    for u in users:
        result[u["username"]] = {
            "name": u["display_name"],
            "category": u["category"],
            "role": u["role"],
            "job_role": u.get("job_role") or ("Master Administrator" if u["role"]=="admin" else "Procurement Manager"),
            "purchaser": u.get("purchaser") or "—",
            "enabled": bool(u.get("enabled",1)),
        }
    return result

# Database is local/self-contained; init also seeds the three demo managers + master admin.
database.init_db(DEFAULT_USERS)
MANAGERS = _refresh_users_from_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# Review actions are same-app deep links. If a browser reload creates a fresh
# Streamlit session, restore the already-selected manager context from the
# review link before the authentication gate, then consume the target below.
_review_user_bootstrap = st.query_params.get("review_user")
if (not st.session_state.authenticated) and _review_user_bootstrap in MANAGERS and st.query_params.get("review_sku"):
    st.session_state.authenticated = True
    st.session_state.username = _review_user_bootstrap

if not st.session_state.authenticated:
    # Minimal, centered, single-screen login. Keep the authenticated application UI untouched.
    _, login_col, _ = st.columns([0.7, 1.6, 0.7], gap="small")
    with login_col:
        st.markdown("""<div class="login-shell">
<div class="login-logo">Procura<span>X</span></div>
<div class="login-sub">AI PROCUREMENT INTELLIGENCE</div>
<div class="login-line"></div>
<div class="login-title">Welcome Back</div>
<div class="login-desc">Sign in to access your procurement dashboard.</div>
""", unsafe_allow_html=True)
        with st.form("procurax_login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username", key="login_username_input")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.checkbox("Remember me", key="login_remember")
            with c2:
                st.markdown('<div class="login-forgot">Forgot password?</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In  →", use_container_width=True)
        if submitted:
            normalized = username.strip().lower()
            user_record = database.authenticate_user(normalized, password)
            if user_record:
                st.session_state.authenticated = True
                st.session_state.username = normalized
                st.session_state.selected_mode = "User Management" if user_record["role"] == "admin" else "Command Center"
                st.session_state.pop("persistence_loaded_for", None)
                for k in ["live_active", "live_materials", "live_suppliers", "live_overrides", "live_analysis_version", "live_previous_decisions", "live_latest_decisions", "live_change_run", "decisions", "approval_workflow", "decision_history", "inventory_receipts", "ai_settings", "analysis_history"]:
                    st.session_state.pop(k, None)
                st.rerun()
            elif database.user_exists(normalized):
                st.error("Invalid password, or this user is disabled. Please contact the Master Admin.")
            else:
                st.error("Username not recognised. Please use your assigned credentials.")
        st.markdown('<div class="login-footer">PREDICT <span>|</span> DECIDE <span>|</span> PROCURE</div></div></div>', unsafe_allow_html=True)
    st.stop()

current_manager = MANAGERS.get(st.session_state.username)
if current_manager is None:
    st.session_state.authenticated = False
    st.rerun()

# -----------------------------
# Data + decision engine
# -----------------------------
@st.cache_data
def make_demo_data():
    """Load the multi-category packaged demo portfolio.

    Refrigerator data is retained as the original 66-SKU portfolio; Air-Conditioning
    and Washing Machine portfolios add 66 SKUs each with independent demand, inventory
    and supplier records for the other manager profiles.
    """
    data_dir = Path(__file__).resolve().parent
    hist_path = data_dir / "sample_procurement_history_all.csv"
    sup_path = data_dir / "sample_supplier_master_all.csv"
    if not hist_path.exists() or not sup_path.exists():
        hist_path = data_dir / "sample_procurement_history.csv"
        sup_path = data_dir / "sample_supplier_master.csv"
    h = pd.read_csv(hist_path, parse_dates=["Date"])
    s = pd.read_csv(sup_path)
    h["Procurement_Category"] = h["Procurement_Category"].fillna("")
    s["Procurement_Category"] = s["Procurement_Category"].fillna("")
    return h, s

DEFAULT_SUPPLIER_WEIGHTS=(35,25,20,20)
DEFAULT_AI_SETTINGS={
    "forecast_model":"Random Forest",
    "forecast_horizon":30,
    "history_window":365,
    "seasonality_enabled":True,
    "trend_enabled":True,
    "confidence_threshold":70,
    "safety_stock_policy":"AI Calculated",
    "safety_stock_days":7,
    "buy_now_days":3,
    "buy_soon_days":7,
    "stockout_protection":True,
    "incoming_po_consideration":True,
    "min_supplier_score":70,
    "low_confidence_review":True,
    "data_anomaly_detection":True,
    "approval_recall_hours":2,
    "po_processing_sla_days":4,
}

FORECAST_MODEL_OPTIONS=["Random Forest","XGBoost","Neural Network","Auto Select"]

@st.cache_data
def supplier_scores(s, weights=DEFAULT_SUPPLIER_WEIGHTS):
    wd,wq,wl,wc=[float(v)/100.0 for v in weights]
    out=[]
    for sku,g in s.groupby("SKU"):
        g=g.copy(); pmin,pmax=g.Quoted_Price.min(),g.Quoted_Price.max(); lmin,lmax=g.Lead_Time_Days.min(),g.Lead_Time_Days.max()
        cost=(1-(g.Quoted_Price-pmin)/max(pmax-pmin,1e-9))*100; lead=(1-(g.Lead_Time_Days-lmin)/max(lmax-lmin,1e-9))*100
        g["Supplier Score"]=(g["On_Time_%"]*wd+g["Quality_%"]*wq+lead*wl+cost*wc).round(1); out.append(g)
    return pd.concat(out,ignore_index=True)

def _forecast_features(series, dates, seasonality=True, trend=True):
    y=pd.Series(series, dtype=float).reset_index(drop=True)
    d=pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    frame=pd.DataFrame({"lag1":y.shift(1),"lag7":y.shift(7),"lag14":y.shift(14),"roll7":y.shift(1).rolling(7).mean(),"roll30":y.shift(1).rolling(30).mean()})
    if seasonality:
        doy=d.dt.dayofyear.fillna(1).to_numpy()
        frame["sin_year"]=np.sin(2*np.pi*doy/365.25)
        frame["cos_year"]=np.cos(2*np.pi*doy/365.25)
    if trend:
        frame["trend"]=np.arange(len(y))/max(len(y)-1,1)
    return frame

def _make_forecast_model(name):
    if name=="XGBoost":
        if XGBRegressor is None:
            raise RuntimeError(
                "XGBoost is not installed in this Python environment. "
                "Please close ProcuraX, launch it using Launch_ProcuraX.command, "
                "or run: python3 -m pip install --user -r requirements.txt"
            )
        return XGBRegressor(n_estimators=80,max_depth=4,learning_rate=0.06,subsample=0.9,colsample_bytree=0.9,objective="reg:squarederror",random_state=42,n_jobs=1,verbosity=0)
    if name=="Neural Network":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,16),activation="relu",solver="adam",alpha=0.001,learning_rate_init=0.01,max_iter=300,early_stopping=True,validation_fraction=0.15,random_state=42))
    return RandomForestRegressor(n_estimators=60,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=1)

def _fit_forecast_model(name, series, dates, horizon, seasonality=True, trend=True, history_window=365):
    y=pd.Series(series, dtype=float).reset_index(drop=True)
    d=pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    window=min(len(y), max(90,int(history_window)))
    y=y.tail(window).reset_index(drop=True); d=d.tail(window).reset_index(drop=True)
    features=_forecast_features(y,d,seasonality,trend)
    train=pd.concat([features,y.rename("target")],axis=1).dropna().reset_index(drop=True)
    if len(train)<45:
        return max(float(y.tail(30).mean()),0.0), None
    x=train.drop(columns=["target"]); target=train["target"]
    model=_make_forecast_model(name); model.fit(x,target)
    hist_y=y.tolist(); future_dates=pd.date_range(d.iloc[-1]+pd.Timedelta(days=1),periods=max(7,min(int(horizon),90)))
    preds=[]
    for fd in future_dates:
        # Append a placeholder for the date being predicted. Lag/rolling features
        # are therefore built only from observed values and earlier recursive predictions.
        temp_dates=list(d)+list(future_dates[:len(preds)+1])
        temp_y=hist_y+preds+[np.nan]
        feat=_forecast_features(temp_y,temp_dates,seasonality,trend).iloc[[-1]]
        pred=max(0.0,float(model.predict(feat)[0])); preds.append(pred)
    return max(float(np.mean(preds)),0.0), model

@st.cache_data(show_spinner=False)
def evaluate_forecast_models(hist, history_window=365, seasonality=True, trend=True):
    names=["Random Forest","XGBoost","Neural Network"]
    rows=[]
    grouped=hist.sort_values("Date").groupby("SKU")
    skus=list(grouped.groups.keys())
    # A small, balanced validation sample keeps Auto Select fast enough for an interactive MVP.
    if len(skus)>6:
        idx=np.linspace(0,len(skus)-1,6,dtype=int); skus=[skus[i] for i in idx]
    for name in names:
        errors=[]; abs_pct=[]
        for sku in skus:
            g=grouped.get_group(sku).sort_values("Date")
            if len(g)<120:
                continue
            val_len=28; train=g.iloc[:-val_len]; val=g.iloc[-val_len:]
            try:
                pred,_=_fit_forecast_model(name,train["Consumption"].values,train["Date"].values,val_len,seasonality,trend,history_window)
                # Validate at the business level using the model's near-term mean forecast.
                actual=float(val["Consumption"].mean()); errors.append(abs(pred-actual)); abs_pct.append(abs(pred-actual)/max(abs(actual),1)*100)
            except Exception:
                continue
        mae=float(np.mean(errors)) if errors else 999999.0
        mape=float(np.mean(abs_pct)) if abs_pct else 999999.0
        rows.append({"Model":name,"MAE":mae,"MAPE":mape,"Validated SKUs":len(errors)})
    return pd.DataFrame(rows).sort_values(["MAE","MAPE"]).reset_index(drop=True)

def _resolve_forecast_model(hist, settings):
    selected=settings.get("forecast_model","Random Forest")
    if selected!="Auto Select":
        return selected, None
    metrics=evaluate_forecast_models(hist,int(settings.get("history_window",365)),bool(settings.get("seasonality_enabled",True)),bool(settings.get("trend_enabled",True)))
    if metrics.empty:
        return "Random Forest", metrics
    return str(metrics.iloc[0]["Model"]), metrics

@st.cache_data(show_spinner=False)
def build_decisions(hist,sup,live_overrides=None,weights=DEFAULT_SUPPLIER_WEIGHTS,settings=None):
    settings=settings or DEFAULT_AI_SETTINGS
    hist=hist.copy(); hist["Date"]=pd.to_datetime(hist["Date"]); sscores=supplier_scores(sup,weights)
    active_model, _metrics=_resolve_forecast_model(hist,settings)
    latest=hist.sort_values("Date").groupby("SKU").tail(1); rows=[]
    horizon=max(7,int(settings.get("forecast_horizon",30)))
    for _,r in latest.iterrows():
        sku=r.SKU; h=hist[hist.SKU==sku].sort_values("Date"); y=h.Consumption.values;
        forecast,_model=_fit_forecast_model(active_model,y,h.Date.values,horizon,bool(settings.get("seasonality_enabled",True)),bool(settings.get("trend_enabled",True)),int(settings.get("history_window",365)))
        if live_overrides is not None and sku in live_overrides:
            forecast=max(0.0,float(live_overrides[sku].get("Consumption",forecast)))
        g=sscores[sscores.SKU==sku]; best=g.sort_values("Supplier Score",ascending=False).iloc[0]
        if float(best["Supplier Score"]) < float(settings.get("min_supplier_score",70)):
            eligible=g[g["Supplier Score"]>=float(settings.get("min_supplier_score",70))]
            if not eligible.empty: best=eligible.sort_values("Supplier Score",ascending=False).iloc[0]
        avg=float(h.Consumption.tail(30).mean()); std=float(h.Consumption.tail(30).std());
        safety=max(avg*float(settings.get("safety_stock_days",7))/max(float(settings.get("buy_soon_days",7)),1),1.65*std*np.sqrt(best.Lead_Time_Days)) if settings.get("safety_stock_policy","AI Calculated")=="AI Calculated" else avg*float(settings.get("safety_stock_days",7))/max(float(settings.get("buy_soon_days",7)),1)
        stock=float(r.Current_Stock)
        if live_overrides is not None and sku in live_overrides:
            stock=max(0.0,float(live_overrides[sku].get("Current_Stock",stock)))
        days=stock/max(forecast,.01); reorder=forecast*best.Lead_Time_Days+safety; target=forecast*max(14,best.Lead_Time_Days)+safety; qty=max(0,int(round(target-stock)))
        pnow=float(h.Unit_Price.tail(7).mean()); pprev=float(h.Unit_Price.tail(30).head(7).mean())
        if live_overrides is not None and sku in live_overrides and "Unit_Price" in live_overrides[sku]:
            pnow=max(0.0,float(live_overrides[sku]["Unit_Price"])); pprev=max(pnow,pprev)
        trend_pct=(pnow-pprev)/max(pprev,.0001)*100
        buy_now=float(settings.get("buy_now_days",3)); buy_soon=float(settings.get("buy_soon_days",7))
        if days < buy_now or (bool(settings.get("stockout_protection",True)) and stock<safety):
            action,priority="BUY NOW","Critical"; reason="Inventory coverage is below the configured immediate-buy threshold or safety-stock requirement."
        elif days < buy_soon or stock<reorder:
            action,priority="BUY SOON","Watch"; reason="Forecast demand and replenishment lead time indicate that an order should be initiated soon."
        elif stock>target*1.5:
            action,priority="DELAY","Normal"; reason="Inventory is above the target range; delaying avoids excess working capital."
        else:
            action,priority="MONITOR","Normal"; reason="Current inventory coverage is adequate under the projected consumption profile."
        if trend_pct>2 and action in ["BUY NOW","BUY SOON"]: reason+=" Recent price movement also supports earlier buying."
        # Savings opportunity uses a competitive benchmark rather than the highest quote.
        # The benchmark is the second-lowest eligible supplier quote, which represents
        # the next-best competitive price available in the current supplier set.
        eligible_quotes=g[g["Supplier Score"]>=float(settings.get("min_supplier_score",70))].copy()
        if eligible_quotes.empty:
            eligible_quotes=g.copy()
        eligible_quotes=eligible_quotes.sort_values("Quoted_Price",ascending=True)
        benchmark_price=float(eligible_quotes.iloc[1]["Quoted_Price"]) if len(eligible_quotes)>=2 else float(best["Quoted_Price"])
        savings=max(0.0,(benchmark_price-float(best["Quoted_Price"]))*qty)
        conf=float(np.clip(70+min(days/30,1)*10+best["Supplier Score"]*.15+abs(trend_pct)*.5,62,98))
        if bool(settings.get("data_anomaly_detection",True)) and float(r.Consumption) > max(avg*2.5, 1):
            reason += " Current consumption is materially above its recent baseline; review the input before approval."
        if bool(settings.get("low_confidence_review",True)) and conf < float(settings.get("confidence_threshold",70)):
            reason += " Confidence is below the configured threshold; manual review is recommended."
        rows.append([sku,r.Material,r.Category,r.get("UOM","unit"),action,priority,stock,forecast,days,safety,qty,best.Supplier,best["Supplier Score"],trend_pct,conf,savings,reason,active_model])
    return pd.DataFrame(rows,columns=["SKU","Material","Category","UOM","Action","Priority","Current Stock","Forecast Daily Demand","Days Cover","Safety Stock","Recommended Qty","Recommended Supplier","Supplier Score","Price Trend %","Confidence %","Potential Savings","Reason","Forecast Model"])

# -----------------------------
# Load data
# -----------------------------
demo_hist,demo_sup=make_demo_data()
SKU_DOMAIN = dict(zip(demo_hist["SKU"].astype(str), demo_hist["Procurement_Category"].astype(str)))
AVAILABLE_CATEGORIES = sorted([x for x in demo_hist["Procurement_Category"].dropna().unique().tolist() if x])

# Sidebar: manager context first, continuous workflow navigation, utilities at the bottom.
st.sidebar.markdown('<div class="brand"><div class="brand-name">Procura<span>X</span></div><div class="brand-sub">AI PROCUREMENT INTELLIGENCE</div></div>',unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="status manager-card" style="margin-bottom:3px;background:#F0FBF9;border-color:#C7EEE7;color:#087F76;"><b>{current_manager["name"]}</b><br><span style="color:#527080;font-weight:600">{current_manager["category"]}</span></div>',unsafe_allow_html=True)
if st.sidebar.button("↪ Sign out", key="sign_out", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.pop("persistence_loaded_for", None)
    st.rerun()
# Real Streamlit spacer: unlike an HTML wrapper, this creates an actual DOM block.
st.sidebar.markdown('<div class="account-separator"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="workflow-heading">Procurement Operations</div>',unsafe_allow_html=True)

# Session state
if "decisions" not in st.session_state: st.session_state.decisions={}
if "live_active" not in st.session_state: st.session_state.live_active=False
if "live_materials" not in st.session_state:
    # Initialize from the demo dataset. The uploaded CSV is loaded below, after the
    # sidebar controls are rendered, so referencing `hist` here would fail on first login.
    _initial_hist = demo_hist.copy() if current_manager.get("role") == "admin" else demo_hist[demo_hist["Procurement_Category"].fillna("") == current_manager["category"]].copy()
    st.session_state.live_materials = (_initial_hist.sort_values("Date").groupby("SKU").tail(1)[["SKU","Material","Category","Procurement_Category","UOM","Consumption","Current_Stock","Unit_Price"]].copy()).reset_index(drop=True)
if "live_suppliers" not in st.session_state:
    _initial_sup = demo_sup.copy() if current_manager.get("role") == "admin" else demo_sup[demo_sup["Procurement_Category"].fillna("") == current_manager["category"]].copy()
    st.session_state.live_suppliers = _initial_sup.reset_index(drop=True)
if "live_overrides" not in st.session_state: st.session_state.live_overrides={}
if "live_analysis_version" not in st.session_state: st.session_state.live_analysis_version=0
if "live_previous_decisions" not in st.session_state: st.session_state.live_previous_decisions=None
if "live_latest_decisions" not in st.session_state: st.session_state.live_latest_decisions=None
if "live_change_run" not in st.session_state: st.session_state.live_change_run=None
if "approval_workflow" not in st.session_state: st.session_state.approval_workflow={}
if "inventory_receipts" not in st.session_state: st.session_state.inventory_receipts={}
if "decision_history" not in st.session_state: st.session_state.decision_history=[]
if "analysis_history" not in st.session_state: st.session_state.analysis_history=[]
if "health_history" not in st.session_state: st.session_state.health_history=[]
if "health_last_signature" not in st.session_state: st.session_state.health_last_signature=None
if "reanalysis_requested" not in st.session_state: st.session_state.reanalysis_requested={}
if "supplier_weights" not in st.session_state: st.session_state.supplier_weights=list(DEFAULT_SUPPLIER_WEIGHTS)
if "supplier_weights_last" not in st.session_state: st.session_state.supplier_weights_last=list(st.session_state.supplier_weights)
if "ai_settings" not in st.session_state: st.session_state.ai_settings=dict(DEFAULT_AI_SETTINGS)

# -----------------------------
# Persistent local SQLite state
# -----------------------------
def _df_records(df):
    if df is None:
        return None
    return df.to_dict(orient="records")

def _df_from_records(records):
    if records is None:
        return None
    return pd.DataFrame(records)

def _persistent_state_payload():
    return {
        "live_active": bool(st.session_state.get("live_active", False)),
        "live_materials": _df_records(st.session_state.get("live_materials")),
        "live_suppliers": _df_records(st.session_state.get("live_suppliers")),
        "live_overrides": st.session_state.get("live_overrides", {}),
        "live_analysis_version": int(st.session_state.get("live_analysis_version", 0)),
        "live_previous_decisions": _df_records(st.session_state.get("live_previous_decisions")),
        "live_latest_decisions": _df_records(st.session_state.get("live_latest_decisions")),
        "live_change_run": st.session_state.get("live_change_run"),
        "decisions": st.session_state.get("decisions", {}),
        "inventory_receipts": st.session_state.get("inventory_receipts", {}),
        "decision_history": st.session_state.get("decision_history", []),
        "analysis_history": st.session_state.get("analysis_history", []),
        "health_history": st.session_state.get("health_history", []),
        "health_last_signature": st.session_state.get("health_last_signature"),
        "reanalysis_requested": st.session_state.get("reanalysis_requested", {}),
        "supplier_weights": list(st.session_state.get("supplier_weights", DEFAULT_SUPPLIER_WEIGHTS)),
        "ai_settings": dict(st.session_state.get("ai_settings", DEFAULT_AI_SETTINGS)),
    }

def persist_current_state():
    try:
        database.save_state(st.session_state.username, _persistent_state_payload())
        database.save_workflow(st.session_state.username, st.session_state.get("approval_workflow", {}))
        st.session_state.pop("persistence_error", None)
    except Exception as exc:
        st.session_state["persistence_error"] = str(exc)

database.init_db()
if st.session_state.get("persistence_loaded_for") != st.session_state.username:
    saved_state = database.load_state(st.session_state.username)
    saved_workflow = database.load_workflow(st.session_state.username)
    if saved_state:
        for key, value in saved_state.items():
            if key in {"live_materials", "live_suppliers", "live_previous_decisions", "live_latest_decisions"}:
                value = _df_from_records(value)
            elif key == "supplier_weights":
                value = list(value)
            elif key == "ai_settings":
                merged=dict(DEFAULT_AI_SETTINGS); merged.update(value or {}); value=merged
            st.session_state[key] = value
    st.session_state.approval_workflow = saved_workflow
    persisted_runs = database.load_analysis_runs(st.session_state.username)
    if persisted_runs:
        st.session_state.analysis_history = persisted_runs
    elif not st.session_state.get("analysis_history"):
        st.session_state.analysis_history = []
    st.session_state.persistence_loaded_for = st.session_state.username

# Streamlit widget state is updated before the script reruns. Sync Settings widgets
# into the decision configuration before the shared decision table is calculated so
# changing a model/policy takes effect immediately, not one interaction later.
def _sync_ai_settings_from_widgets():
    mapping={
        "settings_forecast_model":"forecast_model",
        "settings_forecast_horizon":"forecast_horizon",
        "settings_history_window":"history_window",
        "settings_seasonality":"seasonality_enabled",
        "settings_trend":"trend_enabled",
        "settings_confidence":"confidence_threshold",
        "settings_safety_policy":"safety_stock_policy",
        "settings_safety_days":"safety_stock_days",
        "settings_supplier_min":"min_supplier_score",
        "settings_incoming_po":"incoming_po_consideration",
        "settings_stockout":"stockout_protection",
        "settings_buy_now":"buy_now_days",
        "settings_buy_soon":"buy_soon_days",
        "settings_low_conf":"low_confidence_review",
        "settings_anomaly":"data_anomaly_detection",
        "settings_recall_window":"approval_recall_hours",
        "settings_po_sla":"po_processing_sla_days",
    }
    if st.session_state.get("selected_mode")!="Settings":
        return
    cfg=dict(st.session_state.get("ai_settings",DEFAULT_AI_SETTINGS))
    changed=False
    for widget_key,setting_key in mapping.items():
        if widget_key in st.session_state and st.session_state[widget_key] != cfg.get(setting_key):
            cfg[setting_key]=st.session_state[widget_key]; changed=True
    if changed:
        st.session_state.ai_settings=cfg

_sync_ai_settings_from_widgets()

# When Final Decision is active, use the edited values in the same decision engine.
live_overrides = st.session_state.live_overrides if st.session_state.live_active else None
if st.session_state.live_active:
    sup = st.session_state.live_suppliers.copy()

# Continuous procurement-manager workflow.
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Command Center"

# Direct review links from Command Center. Consume the URL once, then keep the
# target in session state so the exact AI decision can open expanded.
_review_sku = st.query_params.get("review_sku")
_review_action = st.query_params.get("review_action")
if _review_sku:
    st.session_state.review_target_sku = _review_sku
    st.session_state.review_target_action = _review_action or ""
    st.session_state.selected_mode = "AI Decisions"
    st.query_params.clear()

if current_manager.get("role") == "admin":
    nav_items = ["User Management", "Settings"]
else:
    nav_items = ["Command Center", "AI Decisions", "Inventory Intelligence", "Supplier Intelligence", "Negotiation Intelligence", "Procurement Workspace", "What-if Simulator", "Analytics", "Data Hub", "Final Decision", "Settings"]
for item in nav_items:
    active = st.session_state.selected_mode == item
    wrapper = "nav-active" if active else "nav-item"
    st.sidebar.markdown(f'<div class="{wrapper}">', unsafe_allow_html=True)
    if st.sidebar.button(item, key="nav_"+re.sub(r"[^A-Za-z0-9]", "_", item), use_container_width=True):
        # Command Center review deep-links are transient. Any normal sidebar
        # navigation clears the one-time target so it cannot follow the manager
        # around after leaving the reviewed decision.
        st.session_state.pop("review_target_sku", None)
        st.session_state.pop("review_target_action", None)
        st.session_state.pop("final_target_sku", None)
        st.session_state.pop("final_target_action", None)
        st.session_state.selected_mode = item
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

mode = st.session_state.selected_mode

# Low-frequency utilities live after the workflow, with explicit visual separation.
st.sidebar.markdown('<div class="sidebar-bottom-anchor"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-status-card"><b>● AI Decision Engine Active</b><br><span style="color:#6B8197;font-weight:500">Hybrid predictive + rules engine</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="upload-separator"></div>', unsafe_allow_html=True)
uploaded=st.sidebar.file_uploader("Upload procurement CSV", type=["csv"], label_visibility="collapsed")

if uploaded:
    raw=pd.read_csv(uploaded); required={"Date","SKU","Material","Consumption","Current_Stock"}
    if required.issubset(raw.columns):
        hist=raw.copy(); hist["Date"]=pd.to_datetime(hist["Date"], errors="coerce")
        if "Category" not in hist.columns: hist["Category"]="Unclassified"
        if "Unit_Price" not in hist.columns: hist["Unit_Price"]=1.0
        if "UOM" not in hist.columns: hist["UOM"]="unit"
        if "Procurement_Category" not in hist.columns: hist["Procurement_Category"] = hist["SKU"].map(SKU_DOMAIN)
        sup=demo_sup[demo_sup.SKU.isin(hist.SKU.dropna().unique())].copy(); sup=demo_sup.copy() if sup.empty else sup
        if "Procurement_Category" not in sup.columns: sup["Procurement_Category"] = sup["SKU"].map(SKU_DOMAIN)
    else:
        hist,sup=demo_hist.copy(),demo_sup.copy()
else:
    hist,sup=demo_hist.copy(),demo_sup.copy()

# Enforce manager-specific procurement domain. Master Admin can view the full portfolio
# for access-management diagnostics but is not intended to approve procurement.
if current_manager["role"] != "admin":
    hist = hist[hist["Procurement_Category"].fillna("") == current_manager["category"]].copy()
    sup = sup[sup["Procurement_Category"].fillna("") == current_manager["category"]].copy()

# Header helper
def header(title, subtitle, eyebrow="PROCUREMENT INTELLIGENCE"):
    st.markdown(f'<div class="topbar"><div><div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="page-sub">{subtitle}</div></div><div style="text-align:right;color:#6B8197;font-size:.76rem"><b style="color:#173B60">{current_manager["name"]}</b><br>{current_manager["category"]}<br><span style="color:#087F76;font-weight:700">Predict · Decide · Procure</span></div></div>',unsafe_allow_html=True)

# A newly created manager may be assigned to a category whose dataset has not yet
# been configured. Never let an empty domain crash the decision engine; show a clear
# admin-ready state instead.
if current_manager.get("role") != "admin" and hist.empty:
    header("Dataset not configured", "This manager has access to a procurement category, but no dataset is currently configured for it.")
    st.info(f'No procurement dataset is configured for **{current_manager["category"]}**. Ask the Master Admin to assign a configured category or add the category dataset.')
    st.stop()

dec=build_decisions(hist,sup,live_overrides,tuple(st.session_state.supplier_weights),dict(st.session_state.ai_settings))

# Procurement workflow lifecycle. Approved purchase decisions remain recallable for 2 hours,
# then move into PO processing. The 4-day PO processing window is a configurable demo/internal SLA.
APPROVAL_RECALL_HOURS = int(st.session_state.ai_settings.get("approval_recall_hours",2))
PO_PROCESSING_SLA_DAYS = int(st.session_state.ai_settings.get("po_processing_sla_days",4))

def _parse_ts(value):
    try:
        return datetime.fromisoformat(value) if value else None
    except Exception:
        return None

def refresh_workflow_states():
    now=datetime.now()
    changed=False
    for sku,w in list(st.session_state.approval_workflow.items()):
        if not w:
            continue
        status=w.get("status","")
        approved_at=_parse_ts(w.get("approved_at"))
        if status.startswith("Approved — Forwarded") and approved_at:
            if now >= approved_at + timedelta(hours=APPROVAL_RECALL_HOURS):
                w["status"]="PO Processing"
                w["po_processing_started_at"]=(approved_at + timedelta(hours=APPROVAL_RECALL_HOURS)).isoformat()
                changed=True
        if status=="PO Processing":
            started=_parse_ts(w.get("po_processing_started_at"))
            if started and now >= started + timedelta(days=PO_PROCESSING_SLA_DAYS):
                w["status"]="PO Processing — SLA Due"
                w["sla_due_at"]=(started + timedelta(days=PO_PROCESSING_SLA_DAYS)).isoformat()
                changed=True
    if changed:
        persist_current_state()
    return changed

refresh_workflow_states()

def pending_purchase_qty():
    if not bool(st.session_state.ai_settings.get("incoming_po_consideration",True)):
        return {}
    total={}
    for sku,w in st.session_state.approval_workflow.items():
        if not w or int(w.get("qty",0) or 0)<=0:
            continue
        status=w.get("status","")
        if status in {"Received","Closed","Rejected by Manager","Recalled by Manager"}:
            continue
        if status.startswith("Approved — Forwarded") or status.startswith("PO Processing") or status in {"PO Released","In Transit"}:
            total[sku]=total.get(sku,0)+int(w.get("qty",0) or 0)
    return total

def projected_decisions(base):
    out=base.copy()
    pending=pending_purchase_qty()
    for idx,r in out.iterrows():
        add=pending.get(r["SKU"],0)
        if add<=0:
            continue
        stock=float(r["Current Stock"])+add
        days=stock/max(float(r["Forecast Daily Demand"]),.01)
        lead=float(r.get("Recommended Supplier Lead Time",0) or 0)
        # If lead time is not carried in the decision table, derive it from the supplier master.
        sg=sup[(sup.SKU==r["SKU"]) & (sup.Supplier==r["Recommended Supplier"])]
        lead=float(sg["Lead_Time_Days"].iloc[0]) if not sg.empty else 7.0
        if stock<float(r["Safety Stock"]) or days<lead:
            p="Critical"
        elif stock < float(r["Forecast Daily Demand"])*lead + float(r["Safety Stock"]):
            p="Watch"
        else:
            p="Normal"
        out.at[idx,"Projected Stock"]=stock
        out.at[idx,"Projected Days Cover"]=days
        out.at[idx,"Projected Priority"]=p
    return out

def _health_workflow_multiplier(status):
    """Residual exposure after a procurement requirement enters execution."""
    status=str(status or "")
    if status in {"Received","Closed"}: return 0.0
    if status == "PO Released": return 0.40
    if status == "In Transit": return 0.20
    if status.startswith("PO Processing — SLA Due"): return 0.75
    if status.startswith("PO Processing"): return 0.55
    if status.startswith("Approved — Forwarded"): return 0.65
    return 1.0

def procurement_health_details(base, projected=False):
    """Calculate a procurement-risk health score from four business dimensions.

    The score measures underlying procurement exposure, not button clicks. Approval
    reduces exposure progressively as a requirement moves through PO execution; a
    receipt resolves the associated exposure. Values are normalized so a single SKU
    cannot create an implausible score jump.
    """
    if base is None or base.empty:
        return {"score":0,"exposure":0,"inventory":0,"supplier":0,"execution":0,"drivers":[]}
    frame=projected_decisions(base) if projected else base.copy()
    n=max(len(frame),1)

    # 1) Critical procurement exposure (30%): severity × unresolved exposure.
    severity={"BUY NOW":100.0,"BUY SOON":55.0,"DELAY":15.0,"MONITOR":0.0}
    exposure_points=0.0
    exposure_count=0
    for _,r in frame.iterrows():
        action=str(r.get("Action","MONITOR"))
        priority=str(r.get("Priority","Normal"))
        base_risk=severity.get(action,0.0)
        if priority=="Critical": base_risk=min(100.0,base_risk+15.0)
        if base_risk<=0: continue
        exposure_count += 1
        status=st.session_state.approval_workflow.get(str(r["SKU"]),{}).get("status","")
        multiplier=_health_workflow_multiplier(status)
        # If an approved quantity only covers part of the recommendation, keep the
        # uncovered portion exposed.
        rec_qty=max(float(r.get("Recommended Qty",0) or 0),0.0)
        pending_qty=float(pending_purchase_qty().get(str(r["SKU"]),0) or 0) if projected else 0.0
        if pending_qty>0 and rec_qty>0:
            coverage=min(pending_qty/rec_qty,1.0)
            multiplier=max(0.0, multiplier*(1.0-0.35*coverage))
        exposure_points += base_risk*multiplier
    # A healthy portfolio has low unresolved exposure. Multiple unresolved critical
    # decisions accumulate, but the normalization prevents one item from swinging 20+.
    exposure_pressure=(exposure_points/max(n,1))*2.20
    exposure_health=int(round(np.clip(100.0-exposure_pressure,0,100)))

    # 2) Inventory health (30%): evaluate coverage against supplier lead time and
    # safety stock rather than simply counting critical rows.
    inv_scores=[]
    for _,r in frame.iterrows():
        sg=sup[(sup.SKU==r["SKU"]) & (sup.Supplier==r.get("Recommended Supplier","") )]
        lead=float(sg["Lead_Time_Days"].iloc[0]) if not sg.empty else 7.0
        days=float(r.get("Projected Days Cover",r.get("Days Cover",0)) or 0)
        stock=float(r.get("Projected Stock",r.get("Current Stock",0)) or 0)
        safety=float(r.get("Safety Stock",0) or 0)
        if stock < safety: score=25
        elif days < lead: score=45
        elif days < lead+3: score=70
        elif days < lead+7: score=88
        elif days > 90: score=90
        else: score=100
        inv_scores.append(score)
    inventory_health=int(round(float(np.mean(inv_scores)) if inv_scores else 0))

    # 3) Supplier health (20%): existing weighted supplier intelligence.
    scores=supplier_scores(sup,tuple(st.session_state.supplier_weights))
    supplier_health=int(round(float(scores.groupby("SKU")["Supplier Score"].max().reindex(frame.SKU).fillna(70).mean()) if not scores.empty else 70))

    # 4) Execution health (20%): only requirements that have entered workflow count.
    exec_scores=[]
    for _,r in frame.iterrows():
        action=str(r.get("Action","MONITOR"))
        if action not in {"BUY NOW","BUY SOON"}:
            exec_scores.append(100.0); continue
        status=st.session_state.approval_workflow.get(str(r["SKU"]),{}).get("status","")
        if status in {"Received","Closed"}: s=100
        elif status=="PO Released": s=92
        elif status=="In Transit": s=97
        elif status.startswith("PO Processing — SLA Due"): s=55
        elif status.startswith("PO Processing"): s=78
        elif status.startswith("Approved — Forwarded"): s=68
        else: s=35
        exec_scores.append(float(s))
    execution_health=int(round(float(np.mean(exec_scores)) if exec_scores else 100))

    score=int(round(np.clip(exposure_health*.30 + inventory_health*.30 + supplier_health*.20 + execution_health*.20,0,100)))
    drivers=[]
    pending_now=sum(1 for _,r in frame.iterrows() if r.get("Action")=="BUY NOW" and _health_workflow_multiplier(st.session_state.approval_workflow.get(str(r["SKU"]),{}).get("status",""))>0)
    pending_soon=sum(1 for _,r in frame.iterrows() if r.get("Action")=="BUY SOON" and _health_workflow_multiplier(st.session_state.approval_workflow.get(str(r["SKU"]),{}).get("status",""))>0)
    if pending_now: drivers.append(f"{pending_now} BUY NOW requirement{'s' if pending_now!=1 else ''} unresolved")
    if pending_soon: drivers.append(f"{pending_soon} BUY SOON requirement{'s' if pending_soon!=1 else ''} unresolved")
    if inventory_health<75: drivers.append("Inventory coverage is creating stock exposure")
    if supplier_health<75: drivers.append("Supplier performance is below the healthy range")
    if execution_health<75: drivers.append("Procurement execution has unresolved or overdue tasks")
    if not drivers: drivers.append("Procurement exposure is currently well controlled")
    return {"score":score,"exposure":exposure_health,"inventory":inventory_health,"supplier":supplier_health,"execution":execution_health,"drivers":drivers,"unresolved_buy_now":pending_now,"unresolved_buy_soon":pending_soon}

def procurement_health(base, projected=False):
    return procurement_health_details(base, projected=projected)["score"]

critical=int((dec.Priority=="Critical").sum()); watch=int((dec.Priority=="Watch").sum()); savings=float(dec["Potential Savings"].sum())
health_details=procurement_health_details(dec,projected=False)
expected_health_details=procurement_health_details(dec,projected=True)
health=health_details["score"]
expected_health=expected_health_details["score"]

# Persist the latest health snapshot only when the underlying procurement state changes.
# This gives the Command Center a meaningful delta without treating every page rerun as
# a new health event.
def _health_state_signature(details):
    wf=tuple(sorted((str(k),str(v.get("status","")),int(v.get("qty",0) or 0)) for k,v in st.session_state.approval_workflow.items()))
    acts=tuple(sorted((str(r["SKU"]),str(r.get("Action","")),int(float(r.get("Recommended Qty",0) or 0)),str(r.get("Priority",""))) for _,r in dec.iterrows()))
    return repr((acts,wf,tuple(st.session_state.supplier_weights)))

if "health_history" not in st.session_state: st.session_state.health_history=[]
_health_sig=_health_state_signature(health_details)
if st.session_state.get("health_last_signature") != _health_sig:
    st.session_state.health_history.append({"timestamp":datetime.now().isoformat(timespec="seconds"),"score":health,"exposure":health_details["exposure"],"inventory":health_details["inventory"],"supplier":health_details["supplier"],"execution":health_details["execution"],"drivers":health_details["drivers"]})
    st.session_state.health_history=st.session_state.health_history[-50:]
    st.session_state.health_last_signature=_health_sig
    try:
        persist_current_state()
    except Exception:
        pass
component_family_options=["All component families"]+sorted(dec["Category"].dropna().unique().tolist())
priority_rank={"Critical":0,"Watch":1,"Normal":2}

def pending_manager_counts(base):
    """Count only decisions that still require manager action."""
    c=w=0
    for _,r in base.iterrows():
        status=st.session_state.approval_workflow.get(r["SKU"],{}).get("status","")
        if status.startswith("Approved —"):
            continue
        if status in {"Rejected by Manager","Recalled by Manager",""}:
            if r["Priority"]=="Critical": c+=1
            elif r["Priority"]=="Watch": w+=1
    return c,w

# -----------------------------
# Data Hub
# -----------------------------
def _material_overrides(frame):
    """Build the live decision-engine overrides from the currently saved Data Hub inputs.

    This is intentionally derived from the pre-run live material table rather than
    from the historical dataset. The decision engine has a live-input mode, so both
    the before and after snapshots must be evaluated in that same mode. Otherwise the
    first Data Hub run compares the historical forecast against the live-consumption
    forecast and can report many false quantity changes.
    """
    if frame is None or frame.empty:
        return {}
    return {
        str(r["SKU"]): {
            "Consumption": float(r.get("Consumption", 0) or 0),
            "Current_Stock": float(r.get("Current_Stock", 0) or 0),
            "Unit_Price": float(r.get("Unit_Price", 0) or 0),
        }
        for _, r in frame.iterrows()
    }

def _decision_snapshot(frame):
    cols=["SKU","Material","Action","Priority","Recommended Qty","Recommended Supplier","Days Cover","Confidence %"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    x=frame.copy()
    for c in cols:
        if c not in x.columns:
            x[c]=""
    return x[cols].copy()

def build_analysis_change_set(before, after):
    """Compare the immediately preceding live decision set with the current run.

    Only decision-engine outputs that genuinely changed are returned in ``changed``.
    The comparison is SKU-keyed and uses normalized numeric quantities so formatting
    differences cannot masquerade as procurement changes.
    """
    b=_decision_snapshot(before); a=_decision_snapshot(after)
    m=b.merge(a,on="SKU",how="outer",suffixes=("_Before","_After"),indicator=True)
    m["Material"]=m["Material_After"].fillna(m["Material_Before"])
    m["Decision Changed"]=np.where(m["Action_Before"].fillna("").astype(str)!=m["Action_After"].fillna("").astype(str),"YES","NO")
    m["Supplier Changed"]=np.where(m["Recommended Supplier_Before"].fillna("").astype(str)!=m["Recommended Supplier_After"].fillna("").astype(str),"YES","NO")
    q_before=pd.to_numeric(m["Recommended Qty_Before"],errors="coerce")
    q_after=pd.to_numeric(m["Recommended Qty_After"],errors="coerce")
    m["Quantity Changed"]=np.where(q_before.fillna(-1).ne(q_after.fillna(-1)),"YES","NO")
    # New/removed SKUs are genuine changes, but normal demo runs should remain SKU-stable.
    m["Changed"]=(m[["Decision Changed","Supplier Changed","Quantity Changed"]]=="YES").any(axis=1)
    m["Changed Fields"]=m.apply(lambda r:", ".join([x for x,flag in [("Decision",r["Decision Changed"]),("Supplier",r["Supplier Changed"]),("Quantity",r["Quantity Changed"])] if flag=="YES"]) or "No output change",axis=1)
    def driver(r):
        fields=[
            ("Procurement action changed","Decision"),
            ("Recommended supplier changed","Supplier"),
            ("Recommended quantity changed","Quantity"),
        ]
        hits=[label for label,field in fields if r[f"{field} Changed"]=="YES"]
        return " + ".join(hits) if hits else "No decision-engine output changed"
    m["Change Driver"]=m.apply(driver,axis=1)
    changed=m[m["Changed"]].copy()
    return m, changed

def _analysis_run_payload(run_number, timestamp, before, after, settings, changed):
    """Create an immutable, JSON-safe record for one Data Hub analysis run."""
    b=_decision_snapshot(before); a=_decision_snapshot(after)
    merged,_=build_analysis_change_set(b,a)
    changes=[]
    if changed is not None and not changed.empty:
        for _,r in changed.iterrows():
            changes.append({
                "SKU":str(r.get("SKU","")),
                "Material":str(r.get("Material","")),
                "Decision Before":str(r.get("Action_Before","")),
                "Decision After":str(r.get("Action_After","")),
                "Qty Before":float(pd.to_numeric(r.get("Recommended Qty_Before",0),errors="coerce") or 0),
                "Qty After":float(pd.to_numeric(r.get("Recommended Qty_After",0),errors="coerce") or 0),
                "Supplier Before":str(r.get("Recommended Supplier_Before","")),
                "Supplier After":str(r.get("Recommended Supplier_After","")),
                "Days Cover Before":float(pd.to_numeric(r.get("Days Cover_Before",0),errors="coerce") or 0),
                "Days Cover After":float(pd.to_numeric(r.get("Days Cover_After",0),errors="coerce") or 0),
                "Confidence Before":float(pd.to_numeric(r.get("Confidence %_Before",0),errors="coerce") or 0),
                "Confidence After":float(pd.to_numeric(r.get("Confidence %_After",0),errors="coerce") or 0),
                "Changed Fields":str(r.get("Changed Fields","")),
                "Change Driver":str(r.get("Change Driver","")),
            })
    model=str(after["Forecast Model"].mode().iloc[0]) if after is not None and not after.empty and "Forecast Model" in after.columns else str(settings.get("forecast_model","Random Forest"))
    return {
        "run":int(run_number),
        "timestamp":str(timestamp),
        "forecast_model":model,
        "history_window":int(settings.get("history_window",365)),
        "forecast_horizon":int(settings.get("forecast_horizon",30)),
        "materials_analyzed":int(len(a)),
        "decisions_changed":int((merged["Decision Changed"]=="YES").sum()),
        "supplier_changes":int((merged["Supplier Changed"]=="YES").sum()),
        "quantity_changes":int((merged["Quantity Changed"]=="YES").sum()),
        "changes":changes,
        "before_snapshot":_df_records(b),
        "after_snapshot":_df_records(a),
    }

def style_analysis_comparison(display):
    """Apply the agreed procurement status colours and make affected cells obvious."""
    decision_bg={
        "BUY NOW":"#FECACA",
        "BUY SOON":"#FED7AA",
        "MONITOR":"#DCFCE7",
        "DELAY":"#DCFCE7",
    }
    decision_fg={
        "BUY NOW":"#991B1B",
        "BUY SOON":"#9A3412",
        "MONITOR":"#166534",
        "DELAY":"#166534",
    }
    def apply(row):
        styles=pd.Series("",index=row.index)
        before=row.get("Decision Before","")
        after=row.get("Decision After","")
        if before in decision_bg:
            styles["Decision Before"]=f"background-color:{decision_bg[before]};color:{decision_fg[before]};font-weight:700"
        if after in decision_bg:
            styles["Decision After"]=f"background-color:{decision_bg[after]};color:{decision_fg[after]};font-weight:800"
        changed_fields=str(row.get("Changed Fields", ""))
        if "Quantity" in changed_fields:
            styles["Qty Before"]="background-color:#E0F2FE;font-weight:700"
            styles["Qty After"]="background-color:#E0F2FE;font-weight:800"
        if "Supplier" in changed_fields:
            styles["Supplier Before"]="background-color:#F0FDFA;font-weight:700"
            styles["Supplier After"]="background-color:#F0FDFA;font-weight:800"
        if "Decision" in changed_fields:
            styles["Changed Fields"]="background-color:#FEF3C7;font-weight:800"
        return styles
    return display.style.apply(apply,axis=1)

# -----------------------------
# Command Center
# -----------------------------
def render_purchase_task(sku, workflow, current_manager, supplier=None, qty=None, key_suffix="", allow_callback=True):
    """Render an approved purchase task. Callback is intentionally available only in Final Decision."""
    purchaser = workflow.get("purchaser", current_manager["purchaser"])
    supplier = workflow.get("supplier", supplier or "")
    qty = workflow.get("qty", qty if qty is not None else 0)
    st.markdown('<div class="purchase-task-spacer"></div>', unsafe_allow_html=True)
    left, right = st.columns([4.8, 1.0], gap="medium")
    with left:
        st.markdown(
            f'<div class="workflow-task" style="margin:0"><b>PURCHASE TASK</b><br>'
            f'Forwarded to <b>{purchaser}</b> for PO to <b>{supplier}</b>.</div>',
            unsafe_allow_html=True
        )
    with right:
        if allow_callback:
            st.markdown('<div class="callback-col">', unsafe_allow_html=True)
            if st.button("↩ Call Back", key=f"callback_{sku}_{key_suffix}", use_container_width=True):
                st.session_state.decisions[sku] = "Pending"
                st.session_state.approval_workflow[sku] = {
                    "status":"Recalled by Manager",
                    "purchaser":purchaser,
                    "supplier":supplier,
                    "qty":int(qty or 0),
                }
                persist_current_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


if mode=="Command Center":
    header("Command Center","A decision-first view of what ProcuraX thinks needs attention today.")
    cols=st.columns(4)
    pending=pending_purchase_qty()
    pending_critical,pending_watch=pending_manager_counts(dec)
    proj=projected_decisions(dec)
    expected_critical=int((proj["Projected Priority"]=="Critical").sum()) if pending else critical
    expected_watch=int((proj["Projected Priority"]=="Watch").sum()) if pending else watch
    expected_resolved=max(0,critical-expected_critical)
    _health_status = "Healthy" if health>=80 else ("Attention Required" if health>=60 else "At Risk")
    _health_prev = st.session_state.health_history[-2]["score"] if len(st.session_state.health_history)>=2 else health
    _health_delta = health - _health_prev
    _health_delta_text = f"{'+' if _health_delta>=0 else ''}{_health_delta} pts vs previous state" if len(st.session_state.health_history)>=2 else "Current procurement risk state"
    health_tooltip=(
        '<b>Procurement Health</b><br>'
        'Measures current procurement risk and operational readiness.<br><br>'
        '<b>30%</b> Critical Procurement Exposure<br>'
        '<b>30%</b> Inventory Health<br>'
        '<b>20%</b> Supplier Health<br>'
        '<b>20%</b> Execution Health<br><br>'
        'The score follows underlying risk. Approvals improve health only when they reduce unresolved exposure through execution.'
    )
    metrics=[
        ("Procurement Health",f"{health}/100",f"{_health_status} · {_health_delta_text}"),
        ("Pending Manager Decisions",str(pending_critical+pending_watch),f"{pending_critical} critical · {pending_watch} watch"),
        ("Savings Opportunity",f"₹{savings:,.0f}","Potential · competitive benchmark"),
        ("Materials Analysed",str(len(dec)),"Active SKUs")
    ]
    for idx,(c,(lab,val,sub)) in enumerate(zip(cols,metrics)):
        if idx==0:
            c.markdown(f'<div class="metric health-main-card"><div class="label">{lab}</div><div class="value">{val}</div><div class="delta">{sub}</div><div class="health-info">i<span class="health-tooltip">{health_tooltip}</span></div></div>',unsafe_allow_html=True)
        else:
            savings_tip=(
                '<b>Savings Opportunity</b><br>'
                'Estimated procurement savings available from supplier price competition across the current recommendation set.<br><br>'
                '<b>Benchmark:</b> second-lowest eligible supplier quote × recommended purchase quantity.<br><br>'
                '<span style="color:#64748B">This is a potential opportunity, not realized savings.</span>'
            ) if idx==2 else ""
            info=(f'<div class="health-info" style="top:14px;right:14px">i<span class="health-tooltip">{savings_tip}</span></div>' if savings_tip else '')
            c.markdown(f'<div class="metric" style="position:relative"><div class="label">{lab}</div><div class="value">{val}</div><div class="delta">{sub}</div>{info}</div>',unsafe_allow_html=True)
    h1,h2,h3,h4=st.columns(4)
    component_info={
        "Critical Exposure":"Unresolved procurement exposure from BUY NOW/BUY SOON severity, risk level and workflow coverage.",
        "Inventory Health":"Portfolio stock coverage versus supplier lead time, safety stock and excess-inventory exposure.",
        "Supplier Health":"Weighted supplier performance across on-time delivery, quality, lead time and cost competitiveness.",
        "Execution Health":"Progress of required purchases through approval, PO processing, release, transit, receipt and SLA status."
    }
    for c,label,val in zip([h1,h2,h3,h4],["Critical Exposure","Inventory Health","Supplier Health","Execution Health"],[health_details["exposure"],health_details["inventory"],health_details["supplier"],health_details["execution"]]):
        c.markdown(f'<div class="metric" style="margin-top:10px;position:relative"><div class="label">{label}</div><div class="value" style="font-size:1.35rem">{val}/100</div><div class="health-info" style="top:12px;right:12px">i<span class="health-tooltip">{component_info[label]}</span></div></div>',unsafe_allow_html=True)
    if pending:
        st.markdown(f'<div class="insight" style="margin-top:12px"><div class="insight-title">EXPECTED IMPACT FROM APPROVED POs</div><p><b>Current health:</b> {health}/100 → <b>Expected health:</b> {expected_health}/100 after approved POs are received. <b>Current critical inventory risks:</b> {critical} → <b>Expected:</b> {expected_critical} after receipt, meaning <b>{expected_resolved}</b> critical risks are expected to be resolved.</p><p style="margin-top:6px;color:#64748B">Expected values are projections from approved PO quantities. They are not actual inventory and do not change current KPIs until material receipt is recorded.</p></div>',unsafe_allow_html=True)

    alerts=[]
    supplier_score_map=supplier_scores(sup,tuple(st.session_state.supplier_weights)).groupby("SKU")["Supplier Score"].max().to_dict() if not sup.empty else {}
    for _, ar in dec.iterrows():
        sku_sup=sup[sup.SKU==ar.SKU]
        min_lead=float(sku_sup["Lead_Time_Days"].min()) if not sku_sup.empty else 7.0
        approved_qty=int(pending.get(ar["SKU"],0) or 0)
        if float(ar["Days Cover"]) < max(1.0,min_lead):
            if approved_qty>0:
                alerts.append(("MEDIUM","Risk covered by approved PO",f'{ar["Material"]} · {ar.SKU} still has {ar["Days Cover"]:.1f} days cover against a {min_lead:.0f}-day supplier lead time, but {approved_qty:,} {ar.get("UOM","unit")} is approved and expected to address the risk after receipt.'))
            else:
                alerts.append(("HIGH","Stock-out risk",f'{ar["Material"]} · {ar["SKU"]} has {ar["Days Cover"]:.1f} days cover against a {min_lead:.0f}-day supplier lead time.'))
        if float(supplier_score_map.get(ar.SKU,ar["Supplier Score"])) < 60:
            alerts.append(("MEDIUM","Supplier performance",f'{ar["Material"]} · {ar.SKU} has no supplier scoring above 60/100.'))
        if float(ar["Price Trend %"]) >= 8:
            alerts.append(("MEDIUM","Price pressure",f'{ar["Material"]} · {ar["SKU"]} shows {ar["Price Trend %"]:+.1f}% recent price movement.'))
    st.markdown('<div class="section-title">Procurement Alerts & Early Warning</div><div class="section-sub">Signals that may require action before they become procurement problems.</div>',unsafe_allow_html=True)
    if alerts:
        alert_cols=st.columns(min(3,len(alerts)))
        for idx,(ac,(level,title,text)) in enumerate(zip(alert_cols,alerts[:3])):
            # Recover the SKU from the alert text for a direct-review link.
            alert_sku_match=re.search(r"\b[A-Z]{2,4}-\d{3}\b", text)
            alert_sku=alert_sku_match.group(0) if alert_sku_match else ""
            review_action=str(dec.loc[dec["SKU"]==alert_sku,"Action"].iloc[0]) if alert_sku and not dec.loc[dec["SKU"]==alert_sku].empty else "BUY NOW"
            review_user = quote(st.session_state.username or "poojan")
            review_href=f"?review_sku={quote(alert_sku)}&review_action={quote(review_action)}&review_user={review_user}" if alert_sku else "?review_sku="
            pill_class="pill-red" if level=="HIGH" else "pill-amber"
            ac.markdown(f'<div class="panel command-card-shell"><a class="command-review-icon" href="{review_href}" target="_self" aria-label="Review decision">↵</a><span class="pill {pill_class}">{level}</span><div class="ai-title" style="font-size:.92rem">{title}</div><div class="ai-meta" style="padding-right:38px">{text}</div></div>',unsafe_allow_html=True)
        if len(alerts)>3: st.caption(f'{len(alerts)-3} additional alert(s) are present across the active material set.')
    else:
        st.markdown('<div class="insight"><div class="insight-title">✓ NO MATERIAL ALERTS</div><p>ProcuraX is not detecting an immediate stock, supplier or price-pressure warning in the active dataset.</p></div>',unsafe_allow_html=True)
    left,right=st.columns([1.55,1])
    with left:
        priority_actions=dec[dec["Action"].isin(["BUY NOW","BUY SOON"])].copy()
        priority_actions=priority_actions.sort_values(["Priority","Confidence %"],ascending=[True,False])
        st.markdown('<div class="section-title">AI Priority Actions</div><div class="section-sub">The decisions requiring procurement attention right now.</div>',unsafe_allow_html=True)
        if priority_actions.empty:
            monitored_count=int(dec["Action"].isin(["MONITOR","DELAY"]).sum())
            st.markdown(f'<div class="insight"><div class="insight-title">✓ NO IMMEDIATE PROCUREMENT ACTION</div><p>ProcuraX is not recommending an immediate purchase. <b>{monitored_count} materials</b> remain under automated monitoring or delay logic.</p></div>',unsafe_allow_html=True)
        else:
            for _,r in priority_actions.head(4).iterrows():
                cls="critical" if r.Priority=="Critical" else "watch"
                pill="pill-red" if r.Priority=="Critical" else "pill-amber"
                review_user = quote(st.session_state.username or "poojan")
                review_href=f"?review_sku={quote(str(r.SKU))}&review_action={quote(str(r.Action))}&review_user={review_user}"
                st.markdown(f'<div class="ai-card final-decision-card {cls} command-card-shell"><a class="command-review-btn" href="{review_href}" target="_self" aria-label="Review decision">Review</a><span class="pill {pill}">{r.Action}</span><div class="ai-title" style="padding-right:95px">{r.Material} <span style="color:#7C8FA3;font-weight:600">· {r.SKU}</span></div><div class="ai-meta" style="padding-right:95px"><b>Supplier:</b> {r["Recommended Supplier"]} &nbsp; · &nbsp; <b>Qty:</b> {r["Recommended Qty"]:,} {r.get("UOM","unit")} &nbsp; · &nbsp; <b>Confidence:</b> {r["Confidence %"]:.0f}%</div><div class="ai-reason">{r.Reason}</div></div>',unsafe_allow_html=True)
            monitored_count=int(dec["Action"].isin(["MONITOR","DELAY"]).sum())
            if monitored_count:
                st.caption(f"{monitored_count} other material(s) are being monitored automatically and are not shown as priority actions.")
    with right:
        st.markdown('<div class="section-title">Inventory Risk Overview</div>',unsafe_allow_html=True)
        risk=pd.Series({"High Risk":critical,"Medium Risk":watch,"Low Risk":max(0,len(dec)-critical-watch)})
        fig=go.Figure(go.Pie(
            labels=risk.index, values=risk.values, hole=.66, textinfo="none",
            marker=dict(colors=["#EF4444","#F59E0B","#16A34A"], line=dict(color="#FFFFFF", width=3))
        ))
        fig.update_layout(height=290,margin=dict(l=5,r=5,t=5,b=25),showlegend=True,
                          legend=dict(orientation="h",y=-.06,font=dict(size=12)),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<div class="insight"><div class="insight-title">✦ ProcuraX Insight</div><p><b>{critical} materials</b> require immediate review. The decision engine is prioritising stock coverage versus supplier lead time.</p></div>',unsafe_allow_html=True)

# -----------------------------
# AI Decisions
# -----------------------------
elif mode=="AI Decisions":
    header("AI Decision Feed","Actionable procurement recommendations, ranked by urgency and confidence.","AI DECISION ENGINE")
    st.info("AI Decisions is recommendation-only. Review the AI reasoning here; manager approval or rejection is completed in Final Decision.")
    if st.session_state.reanalysis_requested:
        requested=list(st.session_state.reanalysis_requested.keys())
        st.warning(f"Re-analysis requested for {len(requested)} material(s). Review the recommendation below; if input data needs to change, update it in Data Hub and run ProcuraX Analysis.")
    ai_family=st.selectbox("Component family",component_family_options,key="ai_decision_family")
    ai_dec=dec.copy() if ai_family=="All component families" else dec[dec["Category"]==ai_family].copy()
    review_target=st.session_state.get("review_target_sku")
    if review_target and review_target in set(ai_dec["SKU"]):
        target_row=ai_dec[ai_dec["SKU"]==review_target]
        ai_dec=pd.concat([target_row,ai_dec[ai_dec["SKU"]!=review_target]],ignore_index=True)
        st.success(f"Opened directly from Command Center: {target_row.iloc[0]['Material']} · {review_target}")
    t1,t2,t3,t4=st.tabs([f"All ({len(ai_dec)})",f"Buy Now ({(ai_dec.Action=='BUY NOW').sum()})",f"Watch ({(ai_dec.Action=='BUY SOON').sum()})","Monitor / Delay"])
    subsets=[ai_dec,ai_dec[ai_dec.Action=="BUY NOW"],ai_dec[ai_dec.Action=="BUY SOON"],ai_dec[ai_dec.Action.isin(["MONITOR","DELAY"])]]
    for tab_idx,(tab,subset) in enumerate(zip([t1,t2,t3,t4],subsets)):
        with tab:
            for _,r in subset.iterrows():
                cls="critical" if r.Priority=="Critical" else ("watch" if r.Priority=="Watch" else "")
                pill="pill-red" if r.Priority=="Critical" else ("pill-amber" if r.Priority=="Watch" else "pill-green")
                with st.container(border=True, key=f"ai_decision_card_{tab_idx}_{r.SKU}"):
                    a,b,c=st.columns([2.5,1,1])
                    with a:
                        st.markdown(f'<span class="pill {pill}">{r.Action}</span><div class="ai-title">{r.Material} · {r.SKU}</div><div class="ai-meta">{r.Reason}</div>',unsafe_allow_html=True)
                    b.metric("Recommended Qty",f"{r['Recommended Qty']:,} {r.get('UOM','unit')}")
                    b.caption(f"Supplier: {r['Recommended Supplier']}")
                    c.metric("Confidence",f"{r['Confidence %']:.0f}%")
                    c.caption(f"Savings: ₹{r['Potential Savings']:,.0f}")
                    with st.expander("Why did ProcuraX make this decision?", expanded=bool(review_target and r.SKU==review_target)):
                        x,y,z=st.columns(3)
                        sku_sup=sup[sup.SKU==r.SKU]
                        lead_min=sku_sup.Lead_Time_Days.min() if not sku_sup.empty else 0
                        lead_max=sku_sup.Lead_Time_Days.max() if not sku_sup.empty else 0
                        x.metric("Inventory cover",f"{r['Days Cover']:.1f} days",f"Lead time {lead_min:.0f}–{lead_max:.0f}d")
                        y.metric("Forecast demand",f"{r['Forecast Daily Demand']:.1f}/day")
                        z.metric("Supplier score",f"{r['Supplier Score']:.1f}/100")
                        st.markdown(f'<div class="insight"><div class="insight-title">RECOMMENDATION</div><p><b>{r.Action}</b>: order <b>{r["Recommended Qty"]:,} {r.get("UOM","unit")}</b> from <b>{r["Recommended Supplier"]}</b>. This recommendation combines predicted consumption, safety stock, lead time, supplier performance and cost competitiveness.</p></div>',unsafe_allow_html=True)
                        if st.button("Proceed →", key=f"ai_proceed_final_{tab_idx}_{r.SKU}", type="primary", use_container_width=True):
                            st.session_state.final_target_sku = r.SKU
                            st.session_state.final_target_action = r.Action
                            st.session_state.pop("review_target_sku", None)
                            st.session_state.pop("review_target_action", None)
                            st.session_state.selected_mode = "Final Decision"
                            st.rerun()

# -----------------------------
# Procurement workspace
# -----------------------------
elif mode=="Procurement Workspace":
    header("Procurement Workspace","Move from portfolio view to one material and its recommended purchase decision.")
    workspace_family=st.selectbox("Component family",component_family_options,key="workspace_family")
    workspace_dec=dec.copy() if workspace_family=="All component families" else dec[dec["Category"]==workspace_family].copy()
    workspace_skus=workspace_dec.SKU.tolist()
    selected=st.selectbox("Material",workspace_skus,format_func=lambda x:f"{workspace_dec.loc[workspace_dec.SKU==x,'Material'].iloc[0]} · {x}",key="workspace_material")
    r=dec[dec.SKU==selected].iloc[0]
    st.markdown(f'<div class="panel"><span class="pill {"pill-red" if r.Priority=="Critical" else "pill-amber" if r.Priority=="Watch" else "pill-green"}">{r.Action}</span><div class="ai-title" style="font-size:1.35rem">{r.Material} · {r.SKU}</div><div class="ai-meta">AI confidence <b>{r["Confidence %"]:.0f}%</b> · Recommended supplier <b>{r["Recommended Supplier"]}</b> · Potential savings <b>₹{r["Potential Savings"]:,.0f}</b></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Snapshot</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for c,lab,val in [(c1,"Current stock",f"{r['Current Stock']:,.0f}"),(c2,"Forecast / day",f"{r['Forecast Daily Demand']:.1f}"),(c3,"Days cover",f"{r['Days Cover']:.1f}"),(c4,"Recommended order",f"{r['Recommended Qty']:,} {r.get('UOM','unit')}")]: c.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
    pending_item=pending_purchase_qty().get(selected,0)
    if pending_item:
        projected_stock=float(r['Current Stock'])+pending_item
        projected_days=projected_stock/max(float(r['Forecast Daily Demand']),.01)
        st.markdown(f'<div class="insight" style="margin-top:12px"><div class="insight-title">EXPECTED INVENTORY IMPACT</div><p>Current stock <b>{r["Current Stock"]:,.0f} {r.get("UOM","unit")}</b> → expected <b>{projected_stock:,.0f} {r.get("UOM","unit")}</b> after approved PO receipt. Days cover <b>{r["Days Cover"]:.1f}</b> → expected <b>{projected_days:.1f}</b>. Current values remain unchanged until receipt.</p></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Manager Workflow</div><div class="section-sub">Manager approval, rejection and recall actions are intentionally handled only in Final Decision.</div>',unsafe_allow_html=True)
    workflow=st.session_state.approval_workflow.get(selected,{})
    if workflow.get("status"):
        st.markdown(f'<div class="insight workflow-status-card"><div class="insight-title">WORKFLOW STATUS</div><p><b>{workflow["status"]}</b><br>Purchaser: {workflow.get("purchaser",current_manager["purchaser"])} · Supplier: {workflow.get("supplier",r["Recommended Supplier"])} · Qty: {workflow.get("qty",int(r["Recommended Qty"]))} {r.get("UOM","unit")}</p></div>',unsafe_allow_html=True)
        if workflow.get("status", "").startswith("Approved — Forwarded"):
            approved_at=_parse_ts(workflow.get("approved_at"))
            expires=(approved_at + timedelta(hours=APPROVAL_RECALL_HOURS)) if approved_at else datetime.now()
            remaining=max(0,int((expires-datetime.now()).total_seconds()))
            st.markdown(f'<div class="workflow-task"><b>APPROVED · RECALL WINDOW</b><br>Forwarded to <b>{workflow.get("purchaser",current_manager["purchaser"])}</b> for PO to <b>{workflow.get("supplier",r["Recommended Supplier"])}</b>.<br><span style="font-size:.78rem">Recall available for {remaining//60}m {remaining%60:02d}s.</span></div>',unsafe_allow_html=True)
        elif workflow.get("status", "").startswith("PO Processing"):
            started=_parse_ts(workflow.get("po_processing_started_at"))
            elapsed=(datetime.now()-started).total_seconds()/86400 if started else 0
            st.markdown(f'<div class="workflow-task"><b>PO PROCESSING</b><br>Purchaser: <b>{workflow.get("purchaser",current_manager["purchaser"])}</b> · Supplier: <b>{workflow.get("supplier",r["Recommended Supplier"])}</b> · Qty: <b>{int(workflow.get("qty",0)):,} {r.get("UOM","unit")}</b><br><span style="font-size:.78rem">Internal SLA: {min(elapsed,PO_PROCESSING_SLA_DAYS):.1f}/{PO_PROCESSING_SLA_DAYS} days</span></div>',unsafe_allow_html=True)
            if st.button("Mark PO Released",key=f"workspace_release_{selected}",use_container_width=True):
                workflow["status"]="PO Released"; workflow["po_released_at"]=datetime.now().isoformat(); persist_current_state(); st.rerun()
        elif workflow.get("status") == "PO Released":
            st.markdown('<div class="workflow-task"><b>PO RELEASED</b><br>PO has been released. Material receipt is still pending.</div>',unsafe_allow_html=True)
            if st.button("Mark In Transit",key=f"workspace_transit_{selected}",use_container_width=True):
                workflow["status"]="In Transit"; workflow["in_transit_at"]=datetime.now().isoformat(); persist_current_state(); st.rerun()
        elif workflow.get("status") == "In Transit":
            st.markdown('<div class="workflow-task"><b>IN TRANSIT</b><br>Approved quantity is excluded from current inventory until receipt is recorded.</div>',unsafe_allow_html=True)
            if st.button("Mark Material Received",key=f"workspace_receive_{selected}",type="primary",use_container_width=True):
                qty_received=int(workflow.get("qty",0) or 0)
                workflow["status"]="Received"; workflow["received_at"]=datetime.now().isoformat()
                st.session_state.inventory_receipts[selected]=st.session_state.inventory_receipts.get(selected,0)+qty_received
                if selected in st.session_state.live_materials.SKU.values:
                    idx=st.session_state.live_materials.index[st.session_state.live_materials.SKU==selected][0]
                    st.session_state.live_materials.loc[idx,"Current_Stock"] += qty_received
                # Convert the receipt into the active live dataset so the next rerun
                # updates the actual/current KPIs rather than only the expected view.
                st.session_state.live_active=True
                st.session_state.live_overrides={r["SKU"]:{"Consumption":float(r["Consumption"]),"Current_Stock":float(r["Current_Stock"]),"Unit_Price":float(r["Unit_Price"])} for _,r in st.session_state.live_materials.iterrows()}
                workflow["status"]="Closed"
                persist_current_state()
                st.rerun()
        elif workflow.get("status") == "Recalled by Manager":
            st.info("Purchase task recalled by the manager. Go to Final Decision to approve again or reject the recommendation.")
        elif workflow.get("status") == "Rejected by Manager":
            st.warning(f'Recommendation rejected by manager. Reason: {workflow.get("rejection_reason") or "Reason pending"}. Continue the workflow from Final Decision.')
        elif workflow.get("status", "").startswith("Approved"):
            st.markdown(f'<div class="workflow-task"><b>PROCUREMENT ACTION</b><br>{r.Action.title()} confirmed. No PO release is required for this action.</div>',unsafe_allow_html=True)
    else:
        st.info("Awaiting manager review. Go to Final Decision to approve or reject this recommendation.")

    l,rr=st.columns([1.45,1])
    with l:
        h=hist[hist.SKU==selected].sort_values("Date").tail(120)
        fig=px.line(h,x="Date",y="Consumption",title="Consumption trend · last 120 days"); fig.update_layout(height=340,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with rr:
        st.markdown('<div class="section-title">Why this recommendation?</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="insight"><div class="insight-title">1 · PREDICT</div><p>Expected consumption is <b>{r["Forecast Daily Demand"]:.1f} units/day</b>.</p></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="insight" style="margin-top:8px"><div class="insight-title">2 · PROTECT</div><p>Safety stock is <b>{r["Safety Stock"]:.0f}</b>; coverage is <b>{r["Days Cover"]:.1f} days</b>.</p></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="insight" style="margin-top:8px"><div class="insight-title">3 · SELECT</div><p>{r["Recommended Supplier"]} scores <b>{r["Supplier Score"]:.1f}/100</b> across reliability, quality, lead time and cost.</p></div>',unsafe_allow_html=True)

# -----------------------------
# Suppliers
# -----------------------------
elif mode=="Supplier Intelligence":
    header("Supplier Intelligence","Compare supplier performance and see which partner the engine prefers.")

    st.markdown('<div class="section-title">Supplier selection logic</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Set the procurement priority for supplier selection. Changing any weight automatically reallocates the remaining weights so the total always stays at 100%.</div>',unsafe_allow_html=True)

    weight_labels=["On-time delivery","Quality","Lead time","Cost competitiveness"]
    weight_keys=["sw_delivery","sw_quality","sw_lead","sw_cost"]

    def rebalance_supplier_weights(changed_idx):
        keys=weight_keys
        old=list(st.session_state.get("supplier_weights",DEFAULT_SUPPLIER_WEIGHTS))
        new=[int(st.session_state[k]) for k in keys]
        delta=new[changed_idx]-old[changed_idx]
        if delta==0:
            st.session_state.supplier_weights_last=list(new)
            return
        others=[i for i in range(4) if i!=changed_idx]
        step=5
        if delta>0:
            remaining=delta
            # Reduce the largest available weights first, keeping all values in 5% steps.
            while remaining>0:
                candidates=[i for i in others if new[i]>=step]
                if not candidates: break
                i=max(candidates,key=lambda j:new[j])
                new[i]-=step; remaining-=step
        else:
            remaining=-delta
            # Distribute freed weight in 5% increments across the other factors.
            while remaining>0:
                i=min(others,key=lambda j:new[j])
                new[i]+=step; remaining-=step
        new[changed_idx]=int(st.session_state[keys[changed_idx]])
        # Safety normalization in case of an unusual widget-state transition.
        total=sum(new)
        if total!=100:
            new[others[-1]] += 100-total
        for i,k in enumerate(keys):
            st.session_state[k]=int(new[i])
        st.session_state.supplier_weights=list(new)
        st.session_state.supplier_weights_last=list(new)
        persist_current_state()

    # Keep widget keys synchronized with the persistent weighting state.
    for i,k in enumerate(weight_keys):
        if k not in st.session_state:
            st.session_state[k]=int(st.session_state.supplier_weights[i])

    a,b,c,d=st.columns(4)
    for idx,(col,lab,key) in enumerate(zip([a,b,c,d],weight_labels,weight_keys)):
        with col:
            st.selectbox(lab,list(range(0,101,5)),key=key,on_change=rebalance_supplier_weights,args=(idx,),format_func=lambda x:f"{x}%")

    current_weights=tuple(int(st.session_state[k]) for k in weight_keys)
    st.session_state.supplier_weights=list(current_weights)
    total_weight=sum(current_weights)
    st.markdown(f'<div class="insight" style="margin-top:10px"><div class="insight-title">ACTIVE SUPPLIER PRIORITY · {total_weight}%</div><p>On-time {current_weights[0]}% · Quality {current_weights[1]}% · Lead time {current_weights[2]}% · Cost competitiveness {current_weights[3]}%. These weights are now used by the supplier scoring and procurement decision engine.</p></div>',unsafe_allow_html=True)

    scores=supplier_scores(sup,current_weights)
    supplier_family=st.selectbox("Component family",component_family_options,key="supplier_family")
    supplier_dec=dec.copy() if supplier_family=="All component families" else dec[dec["Category"]==supplier_family].copy()
    supplier_skus=supplier_dec.SKU.tolist()
    sku=st.selectbox("Filter by material",["All"]+supplier_skus,format_func=lambda x:"All materials" if x=="All" else f"{supplier_dec.loc[supplier_dec.SKU==x,'Material'].iloc[0]} · {x}",key="supplier_material")
    family_skus=set(supplier_skus)
    view=scores[scores.SKU.isin(family_skus)] if sku=="All" else scores[scores.SKU==sku]
    st.dataframe(view[["SKU","Supplier","On_Time_%","Quality_%","Lead_Time_Days","Quoted_Price","Supplier Score"]].rename(columns={"On_Time_%":"On-time %","Quality_%":"Quality %","Lead_Time_Days":"Lead time (days)","Quoted_Price":"Quoted price"}).sort_values("Supplier Score",ascending=False),use_container_width=True,hide_index=True)

# -----------------------------
# Inventory
# -----------------------------
elif mode=="Inventory Intelligence":
    header("Inventory Intelligence","See where current stock is safe, exposed or approaching a replenishment decision.")

    # Start with a category-level view so the inventory chart remains readable even with a large SKU master.
    categories=["All component families"]+sorted(dec["Category"].dropna().unique().tolist())
    selected_category=st.selectbox("Component family",categories,key="inventory_category")
    filtered_dec=dec.copy() if selected_category=="All component families" else dec[dec["Category"]==selected_category].copy()

    f1,f2=st.columns([1,1])
    with f1:
        priority_filter=st.selectbox("Priority",["All priorities","Critical","Watch","Normal"],key="inventory_priority")
    with f2:
        action_filter=st.selectbox("Recommended action",["All actions","BUY NOW","BUY SOON","MONITOR","DELAY"],key="inventory_action")
    if priority_filter!="All priorities":
        filtered_dec=filtered_dec[filtered_dec["Priority"]==priority_filter]
    if action_filter!="All actions":
        filtered_dec=filtered_dec[filtered_dec["Action"]==action_filter]

    if filtered_dec.empty:
        st.info("No inventory records match the selected filters.")
    else:
        priority_colors={"Critical":"#EF4444","Watch":"#F59E0B","Normal":"#16A34A"}
        with st.container(border=True):
            title_category=selected_category if selected_category!="All component families" else "All component families"
            st.markdown(f'<div class="section-title" style="margin-top:0">Inventory coverage · {title_category}</div>',unsafe_allow_html=True)
            st.caption(f"Showing {len(filtered_dec)} material{'s' if len(filtered_dec)!=1 else ''}. Use the filters above to focus the replenishment view.")
            plot_dec=filtered_dec.sort_values("Days Cover").copy()
            fig=px.bar(plot_dec,x="Material",y="Days Cover",color="Priority",color_discrete_map=priority_colors,text_auto=".1f")
            fig.update_layout(height=380,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",xaxis_title=None,yaxis_title="Days Cover")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        table=filtered_dec[["SKU","Material","Category","UOM","Current Stock","Forecast Daily Demand","Days Cover","Safety Stock","Recommended Qty","Action","Priority"]].copy()
        pending_map=pending_purchase_qty()
        table["Approved PO Qty"]=table["SKU"].map(pending_map).fillna(0).astype(int)
        table["Expected Stock"]=table["Current Stock"]+table["Approved PO Qty"]
        table["Expected Days Cover"]=(table["Expected Stock"]/table["Forecast Daily Demand"].clip(lower=.01)).round(1)
        table=table.sort_values("Days Cover")
        st.dataframe(table,use_container_width=True,hide_index=True)
        if pending_map:
            st.caption("Expected Stock and Expected Days Cover include approved PO quantities, but actual inventory KPIs change only after receipt is recorded.")

# -----------------------------
# Negotiation Intelligence
# -----------------------------
elif mode=="Negotiation Intelligence":
    header("Negotiation Intelligence","Model market cost pressure, supplier price-hike risk and your negotiation leverage before accepting a quote.","NEGOTIATION INTELLIGENCE")

    neg_family=st.selectbox("Component family",component_family_options,key="neg_family")
    neg_dec=dec.copy() if neg_family=="All component families" else dec[dec["Category"]==neg_family].copy()
    n1,n2,n3=st.columns([1.25,1.25,1])
    with n1:
        selected_neg=st.selectbox("Material / SKU",neg_dec.SKU.tolist(),format_func=lambda x:f"{neg_dec.loc[neg_dec.SKU==x,'Material'].iloc[0]} · {x}",key="neg_material")
    nr=neg_dec[neg_dec.SKU==selected_neg].iloc[0]
    sg=sup[sup.SKU==selected_neg].copy()
    supplier_names=sg.Supplier.tolist()
    with n2:
        selected_supplier=st.selectbox("Supplier quote",supplier_names,key="neg_supplier") if supplier_names else "—"
    quote=float(sg.loc[sg.Supplier==selected_supplier,"Quoted_Price"].iloc[0]) if selected_supplier and not sg.empty else float(nr.get("Unit Price",0) or 0)
    with n3:
        annual_volume=st.number_input("Annual purchase volume",min_value=0,step=1000,value=max(10000,int(nr["Recommended Qty"]*12)),key="neg_volume")

    st.markdown('<div class="section-title">External cost drivers</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: metal=st.slider("Metal / element price change",-20,30,0,format="%d%%",key="neg_metal")
    with b: freight=st.slider("Container / shipping change",-20,30,0,format="%d%%",key="neg_freight")
    with c: local_transport=st.slider("Local transport change",-20,30,0,format="%d%%",key="neg_local")
    with d: fx=st.slider("FX / currency change",-10,15,0,format="%d%%",key="neg_fx")

    st.markdown('<div class="section-title">Supplier & buyer leverage</div>',unsafe_allow_html=True)
    e,f,g,h=st.columns(4)
    with e: supplier_history=st.slider("Supplier historical price change",-15,25,3,format="%d%%",key="neg_hist")
    with f: alternatives=st.slider("Qualified alternative suppliers",0,5,min(2,max(0,len(sg)-1)),key="neg_alts")
    with g: dependency=st.slider("Supplier dependency",0,100,40,format="%d%%",key="neg_dep")
    with h: quote_validity=st.slider("Quote validity remaining",1,60,15,format="%d days",key="neg_valid")

    # Explainable cost-driver model. We intentionally keep this deterministic and auditable in the MVP.
    market_pressure=(metal*0.50)+(freight*0.20)+(local_transport*0.15)+(fx*0.15)
    supplier_pressure=(market_pressure*0.70)+(supplier_history*0.30)
    buyer_leverage=min(100,max(0,alternatives*18 + min(annual_volume/25000,4)*12 + (100-dependency)*0.25 + min(quote_validity/30,1)*5))
    hike_risk=min(100,max(0,50 + supplier_pressure*2.0 + dependency*0.35 - alternatives*9 - min(quote_validity,30)*0.35))

    # Translate market pressure into a fair-price band. Positive pressure supports a higher fair price; negative pressure creates negotiation headroom.
    fair_price=quote*(1+supplier_pressure/100*0.60)
    target_discount=min(0.08,max(0.015,buyer_leverage/100*0.055 + max(0,-supplier_pressure)/100*0.025))
    target_price=max(0, fair_price*(1-target_discount))
    walkaway=max(target_price, fair_price*(1+0.02 + max(0,supplier_pressure)/100*0.05))

    if buyer_leverage>=65 and supplier_pressure<=5:
        position="NEGOTIATION IS IN OUR FAVOUR"; pos_class="pill-green"
    elif buyer_leverage>=45 and supplier_pressure<=12:
        position="NEGOTIATE WITH CAUTION"; pos_class="pill-amber"
    else:
        position="AVOID NEGOTIATION"; pos_class="pill-red"

    if hike_risk>=65: hike_label="HIGH"; hike_class="pill-red"
    elif hike_risk>=40: hike_label="MEDIUM"; hike_class="pill-amber"
    else: hike_label="LOW"; hike_class="pill-green"

    if position=="NEGOTIATION IS IN OUR FAVOUR":
        action_text="Enter negotiation now. Lead with the reduction in external cost pressure and use alternative suppliers and volume as leverage."
    elif position=="NEGOTIATE WITH CAUTION":
        action_text="Negotiate selectively. Protect supply continuity and anchor the discussion around verified cost drivers rather than a hard price cut."
    else:
        action_text="Avoid aggressive negotiation. Consider securing supply or locking the current quote if market and dependency signals remain unfavourable."

    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f'<div class="metric negotiation-metric" style="min-height:140px;height:140px"><div class="label">Negotiation position</div><div class="value" style="font-size:1.30rem;line-height:1.15">{position}</div><div class="delta">Buyer leverage {buyer_leverage:.0f}/100</div></div>',unsafe_allow_html=True)
    k2.markdown(f'<div class="metric negotiation-metric" style="min-height:140px;height:140px"><div class="label">Supplier price-hike risk</div><div class="value" style="font-size:1.80rem">{hike_risk:.0f}%</div><div class="delta"><span class="pill {hike_class}">{hike_label}</span></div></div>',unsafe_allow_html=True)
    k3.markdown(f'<div class="metric negotiation-metric" style="min-height:140px;height:140px"><div class="label">Recommended target</div><div class="value">₹{target_price:,.2f}</div><div class="delta">Current quote ₹{quote:,.2f}</div></div>',unsafe_allow_html=True)
    k4.markdown(f'<div class="metric negotiation-metric" style="min-height:140px;height:140px"><div class="label">Walk-away price</div><div class="value">₹{walkaway:,.2f}</div><div class="delta">Estimated fair price ₹{fair_price:,.2f}</div></div>',unsafe_allow_html=True)

    left,right=st.columns([1.15,1])
    with left:
        st.markdown('<div class="section-title">ProcuraX negotiation assessment</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><span class="pill {pos_class}">{position}</span><div class="ai-title">{nr["Material"]} · {selected_neg}</div><div class="ai-meta"><b>Supplier:</b> {selected_supplier} &nbsp; · &nbsp; <b>Quote:</b> ₹{quote:,.2f} &nbsp; · &nbsp; <b>Cost pressure:</b> {supplier_pressure:+.1f}%</div><div class="ai-reason">{action_text}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">Negotiation rationale</div>',unsafe_allow_html=True)
        reasons=[]
        if market_pressure<0: reasons.append(f"External cost drivers imply a {abs(market_pressure):.1f}% reduction in supplier cost pressure.")
        elif market_pressure>0: reasons.append(f"External cost drivers imply a {market_pressure:.1f}% increase in supplier cost pressure.")
        else: reasons.append("External cost drivers are broadly neutral.")
        reasons.append(f"Buyer leverage is {buyer_leverage:.0f}/100 based on alternatives, purchase volume, dependency and quote validity.")
        reasons.append(f"Supplier price-hike risk is {hike_risk:.0f}% based on market pressure, historical pricing behaviour and switching leverage.")
        st.markdown('<div class="insight"><div class="insight-title">WHY THIS MATTERS</div><p>• '+'<br>• '.join(reasons)+'</p></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Negotiation range</div>',unsafe_allow_html=True)
        range_df=pd.DataFrame({"Price Point":["Recommended target","Current quote","Walk-away"],"Price":[target_price,quote,walkaway]})
        fig=px.bar(range_df,x="Price Point",y="Price",text_auto=".2f",title="Target vs current quote")
        fig.update_layout(height=290,margin=dict(l=10,r=10,t=50,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        saving=max(0,(quote-target_price))*max(0,annual_volume)
        st.markdown(f'<div class="insight"><div class="insight-title">POTENTIAL ANNUAL SAVING</div><p><b>₹{saving:,.0f}</b> if the target price is achieved across the stated annual volume.</p></div>',unsafe_allow_html=True)

    # Negotiation guidance is presented through compact dropdowns so both guidance
    # areas remain available without cluttering the page.
    concessions=["Volume commitment","Longer contract / repeat business","Payment-term improvement","Forecast visibility"]

    st.markdown('<div class="section-title">How you can proceed with the negotiation</div>',unsafe_allow_html=True)
    if buyer_leverage >= 65:
        anchor=max(0,target_price*0.97)
        leverage_points=[
            "Use alternative suppliers as credible switching leverage.",
            "Use annual volume as the primary commercial trade-off.",
            "Ask for price protection or a validity extension before offering concessions."
        ]
        timing="Negotiate now"
        negotiation_note="Current conditions support an active negotiation."
    elif buyer_leverage >= 45:
        anchor=max(0,target_price*0.985)
        leverage_points=[
            "Lead with verified external cost drivers.",
            "Trade volume or payment terms only for a measurable price concession.",
            "Keep at least one qualified alternative supplier in the discussion."
        ]
        timing="Negotiate selectively"
        negotiation_note="Negotiation is possible, but supply continuity and commercial justification should remain protected."
    else:
        anchor=max(0,target_price)
        leverage_points=[]
        timing="Do not use an aggressive price-cut approach"
        negotiation_note="Current conditions do not support an aggressive negotiation. Focus on supply continuity and commercial protection instead."

    with st.expander("▸ Recommended negotiation playbook", expanded=False):
        if position in ["NEGOTIATION IS IN OUR FAVOUR","NEGOTIATE WITH CAUTION"]:
            pb1,pb2=st.columns(2)
            with pb1:
                st.markdown(f'<div class="panel"><div class="ai-title">Negotiation playbook</div><div class="ai-meta"><b>Opening anchor:</b> ₹{anchor:,.2f}<br><b>Target:</b> ₹{target_price:,.2f}<br><b>Walk-away:</b> ₹{walkaway:,.2f}<br><b>Timing:</b> {timing}</div></div>',unsafe_allow_html=True)
            with pb2:
                st.markdown('<div class="panel"><div class="ai-title">What to trade — and what not to give away</div><div class="ai-meta"><b>Possible concessions:</b> '+', '.join(concessions)+f'<br><br><b>Do not cross:</b> ₹{walkaway:,.2f} per unit without a new commercial justification.</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="insight" style="margin-top:10px"><div class="insight-title">USE THESE LEVERS</div><p>• '+'<br>• '.join(leverage_points)+'</p></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight"><div class="insight-title">NEGOTIATION STATUS</div><p>{negotiation_note}</p></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">How to safeguard against a supplier price hike</div>',unsafe_allow_html=True)
    with st.expander("▸ Price-hike safeguards", expanded=False):
        if hike_risk >= 65:
            safeguards=[
                "Request a transparent cost breakdown before accepting any increase.",
                "Lock the current price or agree a capped increase with a defined validity period.",
                "Do not accept urgency-based price hikes without supporting market evidence.",
                "Keep qualified alternate suppliers active to reduce switching dependency."
            ]
            safeguard_intro="Price-hike risk is high. Prioritise contractual and sourcing protections before accepting any increase."
        elif hike_risk >= 40:
            safeguards=[
                "Ask the supplier to substantiate any proposed increase with cost-driver evidence.",
                "Request a price-validity commitment and avoid open-ended quote expiry.",
                "Keep at least one alternative supplier commercially warm.",
                "Trade any concession only for a measurable commercial benefit."
            ]
            safeguard_intro="Price-hike risk is moderate. Keep commercial protections active while monitoring the supplier's cost justification."
        else:
            safeguards=[
                "Keep the quote validity and agreed commercial terms documented.",
                "Monitor major cost drivers before accepting an unexpected increase.",
                "Maintain at least one qualified alternate supplier as a contingency.",
                "Ask for evidence before accepting any supplier-initiated price change."
            ]
            safeguard_intro="Price-hike risk is currently low, but these controls help prevent avoidable increases."
        st.markdown(f'<div class="insight"><div class="insight-title">PRICE-HIKE SAFEGUARDS</div><p>{safeguard_intro}</p><p>• '+'<br>• '.join(safeguards)+'</p></div>',unsafe_allow_html=True)

# -----------------------------
# What-if
# -----------------------------
elif mode=="What-if Simulator":
    header("What-if Simulator","Stress-test procurement decisions before market or operational conditions change.","SCENARIO INTELLIGENCE")
    whatif_family=st.selectbox("Component family",component_family_options,key="whatif_family")
    whatif_dec=dec.copy() if whatif_family=="All component families" else dec[dec["Category"]==whatif_family].copy()
    st.markdown('<div class="panel"><div class="section-title" style="margin-top:0">What happens if conditions change?</div>',unsafe_allow_html=True)
    pcol,dcol,lcol=st.columns(3)
    with pcol: price=st.slider("Commodity price increase",0,30,10,format="%d%%")
    with dcol: demand=st.slider("Demand increase",0,50,15,format="%d%%")
    with lcol: delay=st.slider("Additional supplier lead time",0,15,3,format="%d days")
    sim=whatif_dec.copy(); sim["Scenario Demand"]=sim["Forecast Daily Demand"]*(1+demand/100); sim["Scenario Days Cover"]=sim["Current Stock"]/sim["Scenario Demand"].replace(0,np.nan); sim["Price Exposure"]=sim["Recommended Qty"]*price/100
    impacted=sim[(sim["Scenario Days Cover"]<sim["Days Cover"]+delay)|sim.Action.isin(["BUY NOW","BUY SOON"])]
    st.markdown('</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown(f'<div class="metric"><div class="label">Materials requiring review</div><div class="value">{len(impacted)}</div><div class="delta">Under this scenario</div></div>',unsafe_allow_html=True)
    b.markdown(f'<div class="metric"><div class="label">Additional price exposure</div><div class="value">₹{impacted["Price Exposure"].sum():,.0f}</div><div class="delta">If prices rise {price}%</div></div>',unsafe_allow_html=True)
    c.markdown(f'<div class="metric"><div class="label">Demand stress</div><div class="value">+{demand}%</div><div class="delta">Lead time stress +{delay} days</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario impact</div>',unsafe_allow_html=True)
    st.dataframe(impacted[["SKU","Material","Scenario Demand","Scenario Days Cover","Action","Recommended Supplier"]].sort_values("Scenario Days Cover"),use_container_width=True,hide_index=True)

# -----------------------------
# Analytics
# -----------------------------
elif mode=="Analytics":
    header("Procurement Analytics","Portfolio-level performance indicators and recommendation mix.")
    analytics_family=st.selectbox("Component family",component_family_options,key="analytics_family")
    analytics_dec=dec.copy() if analytics_family=="All component families" else dec[dec["Category"]==analytics_family].copy()
    analytics_critical=int((analytics_dec.Priority=="Critical").sum())
    a,b,c,d=st.columns(4)
    vals=[("Average supplier score",f"{analytics_dec['Supplier Score'].mean():.1f}/100"),("Average confidence",f"{analytics_dec['Confidence %'].mean():.1f}%"),("Potential savings",f"₹{analytics_dec['Potential Savings'].sum():,.0f}"),("Critical share",f"{analytics_critical/len(analytics_dec)*100:.0f}%")]
    for col,(lab,val) in zip([a,b,c,d],vals): col.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
    l,r=st.columns(2)
    with l:
        fig=px.pie(analytics_dec,names="Action",title="AI recommendation mix",hole=.52); fig.update_layout(height=330,margin=dict(l=10,r=10,t=50,b=10),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with r:
        fig=px.scatter(analytics_dec,x="Days Cover",y="Supplier Score",size="Recommended Qty",color="Priority",hover_name="Material",title="Coverage vs supplier strength"); fig.update_layout(height=330,margin=dict(l=10,r=10,t=50,b=10),paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.dataframe(analytics_dec.assign(_priority_rank=analytics_dec["Priority"].map(priority_rank)).sort_values(["_priority_rank","Confidence %"],ascending=[True,False]).drop(columns=["_priority_rank"]),use_container_width=True,hide_index=True)

# -----------------------------
# Data Hub
# -----------------------------
elif mode=="Data Hub":
    header("Data Hub","Edit procurement and supplier inputs here, then run ProcuraX analysis.","DATA & INTEGRATION")
    st.markdown('<div class="section-sub" style="margin-top:-4px;margin-bottom:18px">Edit procurement and supplier inputs below, then run the same ProcuraX decision engine to refresh the active recommendation set.</div>',unsafe_allow_html=True)

    a,b,c=st.columns(3)
    a.markdown(f'<div class="metric"><div class="label">Active materials</div><div class="value">{len(st.session_state.live_materials)}</div><div class="delta">Editable live inputs</div></div>',unsafe_allow_html=True)
    b.markdown(f'<div class="metric"><div class="label">Supplier records</div><div class="value">{len(st.session_state.live_suppliers)}</div><div class="delta">Editable supplier master</div></div>',unsafe_allow_html=True)
    c.markdown(f'<div class="metric"><div class="label">Analysis runs</div><div class="value">{st.session_state.live_analysis_version}</div><div class="delta">Same decision engine</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">1 · Live material inputs</div>',unsafe_allow_html=True)
    st.caption("Filter the material master first, then edit Consumption, Current Stock or Unit Price. These values are fed into the active procurement decision engine.")
    data_family_options=["All component families"]+sorted(st.session_state.live_materials["Category"].dropna().unique().tolist())
    data_family=st.selectbox("Component family",data_family_options,key="datahub_family")
    lm=st.session_state.live_materials.copy()
    if data_family!="All component families":
        lm=lm[lm["Category"]==data_family].copy()
    data_material_options=["All materials"]+lm["SKU"].tolist()
    data_material=st.selectbox("Material",data_material_options,format_func=lambda x:"All materials" if x=="All materials" else f"{lm.loc[lm.SKU==x,'Material'].iloc[0]} · {x}",key="datahub_material")
    if data_material!="All materials":
        lm=lm[lm["SKU"]==data_material].copy()
    edited_materials=st.data_editor(lm,use_container_width=True,hide_index=True,num_rows="fixed",key=f"live_material_editor_{data_family}_{data_material}",column_config={
        "SKU":st.column_config.TextColumn("SKU",disabled=True),
        "Material":st.column_config.TextColumn("Material",disabled=True),
        "Category":st.column_config.TextColumn("Category",disabled=True),
        "Procurement_Category":st.column_config.TextColumn("Manager Domain",disabled=True),
        "UOM":st.column_config.TextColumn("UOM",disabled=True),
        "Consumption":st.column_config.NumberColumn("Daily Consumption",min_value=0.0,step=1.0),
        "Current_Stock":st.column_config.NumberColumn("Current Stock",min_value=0.0,step=1.0),
        "Unit_Price":st.column_config.NumberColumn("Unit Price",min_value=0.0,step=0.01)})

    st.markdown('<div class="section-title">2 · Live supplier inputs</div>',unsafe_allow_html=True)
    st.caption("Change supplier price, delivery reliability, quality or lead time to demonstrate supplier selection changing live.")
    ls=st.session_state.live_suppliers.copy()
    edited_suppliers=st.data_editor(ls,use_container_width=True,hide_index=True,num_rows="fixed",key="live_supplier_editor",column_config={
        "SKU":st.column_config.TextColumn("SKU",disabled=True),
        "Supplier_ID":st.column_config.TextColumn("ID",disabled=True),
        "Supplier":st.column_config.TextColumn("Supplier",disabled=True),
        "Procurement_Category":st.column_config.TextColumn("Manager Domain",disabled=True),
        "On_Time_%":st.column_config.NumberColumn("On-Time %",min_value=0.0,max_value=100.0,step=1.0),
        "Quality_%":st.column_config.NumberColumn("Quality %",min_value=0.0,max_value=100.0,step=1.0),
        "Lead_Time_Days":st.column_config.NumberColumn("Lead Time (days)",min_value=0.0,step=1.0),
        "Quoted_Price":st.column_config.NumberColumn("Quoted Price",min_value=0.0,step=0.01)})

    st.markdown("---")
    x,y=st.columns([1,1])
    with x:
        if st.button("⚡ Run ProcuraX Analysis",type="primary",use_container_width=True):
            mat=st.session_state.live_materials.copy()
            if not edited_materials.empty:
                mat=mat.set_index("SKU")
                ed=edited_materials.set_index("SKU")
                for col in ["Consumption","Current_Stock","Unit_Price"]:
                    if col in ed.columns:
                        mat.loc[ed.index,col]=ed[col]
                mat=mat.reset_index()
            ss=edited_suppliers.copy()
            numeric=["Consumption","Current_Stock","Unit_Price"]
            for col in numeric: mat[col]=pd.to_numeric(mat[col],errors="coerce").fillna(0).clip(lower=0)
            for col in ["On_Time_%","Quality_%","Lead_Time_Days","Quoted_Price"]: ss[col]=pd.to_numeric(ss[col],errors="coerce").fillna(0).clip(lower=0)
            ss["On_Time_%"]=ss["On_Time_%"].clip(0,100); ss["Quality_%"]=ss["Quality_%"].clip(0,100)
            # Capture the exact live-input decision state immediately before this run.
            # IMPORTANT: the first run must use the same live-input mode as the edited
            # run. Falling back to the historical forecast here makes unchanged SKUs
            # appear to have quantity changes on the first analysis.
            before_materials=st.session_state.live_materials.copy()
            before_suppliers=st.session_state.live_suppliers.copy()
            before_overrides=_material_overrides(before_materials)
            before_dec=build_decisions(
                hist,
                before_suppliers,
                before_overrides,
                tuple(st.session_state.supplier_weights),
                dict(st.session_state.ai_settings),
            )
            st.session_state.live_materials=mat
            st.session_state.live_suppliers=ss
            new_overrides=_material_overrides(mat)
            st.session_state.live_overrides=new_overrides
            # Build the new decision set only after all edited inputs have been applied.
            new_dec=build_decisions(hist,ss,new_overrides,tuple(st.session_state.supplier_weights),dict(st.session_state.ai_settings))
            st.session_state.live_previous_decisions=before_dec[["SKU","Material","Action","Priority","Recommended Qty","Recommended Supplier","Days Cover","Confidence %"]].copy()
            st.session_state.live_latest_decisions=new_dec[["SKU","Material","Action","Priority","Recommended Qty","Recommended Supplier","Days Cover","Confidence %"]].copy()
            st.session_state.live_active=True
            st.session_state.live_analysis_version += 1
            st.session_state.live_change_run=st.session_state.live_analysis_version
            # A new dataset analysis invalidates prior approvals/rejections because the recommendation may have changed.
            st.session_state.approval_workflow = {}
            st.session_state.decisions = {}
            st.session_state.reanalysis_requested = {}
            run_ts=datetime.now().strftime("%d %b %Y · %H:%M")
            run_payload=_analysis_run_payload(st.session_state.live_analysis_version,run_ts,before_dec,new_dec,dict(st.session_state.ai_settings),changed_dec if "changed_dec" in locals() else build_analysis_change_set(st.session_state.live_previous_decisions,st.session_state.live_latest_decisions)[1])
            st.session_state.analysis_history.insert(0,run_payload)
            st.session_state.analysis_history=st.session_state.analysis_history[:50]
            database.save_analysis_run(st.session_state.username,st.session_state.live_analysis_version,run_payload)
            st.session_state.decision_history.append({"run":st.session_state.live_analysis_version,"timestamp":run_ts,"open_decisions":int(new_dec["Action"].isin(["BUY NOW","BUY SOON"]).sum()),"critical":int((new_dec["Priority"]=="Critical").sum()),"savings":float(new_dec["Potential Savings"].sum()),"forecast_model":run_payload["forecast_model"]})
            st.session_state.decision_history=st.session_state.decision_history[-50:]
            st.success("Analysis rerun. ProcuraX is now using your edited live data.")
            persist_current_state()
            st.rerun()
    with y:
        if st.button("↩ Reset to Demo Dataset",use_container_width=True):
            st.session_state.live_active=False
            st.session_state.live_materials=(demo_hist[demo_hist["Procurement_Category"]==current_manager["category"]].sort_values("Date").groupby("SKU").tail(1)[["SKU","Material","Category","Procurement_Category","UOM","Consumption","Current_Stock","Unit_Price"]].copy()).reset_index(drop=True)
            st.session_state.live_suppliers=demo_sup.copy()
            st.session_state.live_overrides={}
            st.session_state.live_previous_decisions=None
            st.session_state.live_latest_decisions=None
            st.session_state.live_change_run=None
            st.session_state.live_analysis_version=0
            st.session_state.approval_workflow={}
            st.session_state.inventory_receipts={}
            st.session_state.decisions={}
            st.session_state.reanalysis_requested={}
            st.session_state.decision_history=[]
            st.session_state.analysis_history=[]
            st.session_state.health_history=[]
            st.session_state.health_last_signature=None
            database.clear_analysis_runs(st.session_state.username)
            persist_current_state()
            st.rerun()
    st.markdown('<div class="section-title">3 · Latest analysis impact</div>',unsafe_allow_html=True)
    latest_before=st.session_state.get("live_previous_decisions")
    latest_after=st.session_state.get("live_latest_decisions")
    if latest_before is not None and latest_after is not None:
        merged_latest, changed_latest=build_analysis_change_set(latest_before,latest_after)
        a,b,c,d=st.columns(4)
        a.markdown(f'<div class="metric"><div class="label">Analysis Run</div><div class="value">#{st.session_state.live_change_run}</div><div class="delta">Latest live dataset</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="metric"><div class="label">Decisions Changed</div><div class="value">{int((merged_latest["Decision Changed"]=="YES").sum())}</div><div class="delta">Action category changed</div></div>',unsafe_allow_html=True)
        c.markdown(f'<div class="metric"><div class="label">Supplier Changes</div><div class="value">{int((merged_latest["Supplier Changed"]=="YES").sum())}</div><div class="delta">Recommended supplier changed</div></div>',unsafe_allow_html=True)
        d.markdown(f'<div class="metric"><div class="label">Quantity Changes</div><div class="value">{int((merged_latest["Quantity Changed"]=="YES").sum())}</div><div class="delta">Recommended quantity changed</div></div>',unsafe_allow_html=True)
        if not changed_latest.empty:
            st.caption("Only outputs affected by the latest Data Hub run are shown. This comparison is reset and rebuilt on every analysis run.")
            display=changed_latest[["SKU","Material","Action_Before","Action_After","Recommended Qty_Before","Recommended Qty_After","Recommended Supplier_Before","Recommended Supplier_After","Days Cover_Before","Days Cover_After","Confidence %_Before","Confidence %_After","Changed Fields","Change Driver"]].copy()
            display.columns=["SKU","Material","Decision Before","Decision After","Qty Before","Qty After","Supplier Before","Supplier After","Days Cover Before","Days Cover After","Confidence Before","Confidence After","Changed Fields","Change Driver"]
            st.dataframe(style_analysis_comparison(display),use_container_width=True,hide_index=True)
        else:
            st.success("The latest analysis did not change any procurement decision, supplier recommendation or recommended quantity.")
    else:
        st.info("Edit the data above and click Run ProcuraX Analysis. The latest before/after impact will appear here.")

    st.markdown('<div class="section-title">4 · Analysis history</div>',unsafe_allow_html=True)
    st.caption("Every Data Hub run is retained as an immutable comparison. The latest run remains above; older runs can be expanded below.")
    if st.session_state.get("analysis_history"):
        for run in st.session_state.analysis_history:
            changes=run.get("changes",[])
            label=f"Analysis #{run.get('run')} · {run.get('timestamp')} · {len(changes)} change{'s' if len(changes)!=1 else ''} · {run.get('forecast_model','Random Forest')}"
            with st.expander(label, expanded=False):
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Materials analyzed",run.get("materials_analyzed",0))
                m2.metric("Decisions changed",run.get("decisions_changed",0))
                m3.metric("Supplier changes",run.get("supplier_changes",0))
                m4.metric("Quantity changes",run.get("quantity_changes",0))
                if changes:
                    hd=pd.DataFrame(changes)
                    st.dataframe(style_analysis_comparison(hd.rename(columns={
                        "Decision Before":"Decision Before","Decision After":"Decision After",
                        "Qty Before":"Qty Before","Qty After":"Qty After",
                    })),use_container_width=True,hide_index=True)
                else:
                    st.success("No procurement output changed in this analysis run.")
    else:
        st.info("No historical analysis runs recorded yet.")

    st.markdown('<div class="section-title">5 · Analysis status</div>',unsafe_allow_html=True)
    if st.session_state.live_active:
        st.success(f"Live analysis #{st.session_state.live_analysis_version} is active. Final Decision uses the same latest before/after comparison.")
    else:
        st.info("Edit the data above and click Run ProcuraX Analysis. Final Decision will then show the change summary.")
    st.markdown('<div class="section-title">CSV upload</div>',unsafe_allow_html=True)
    st.caption("CSV upload remains available in the sidebar for larger datasets. After running an analysis, use Final Decision for the before-versus-after summary.")

# -----------------------------
# Final Decision
# -----------------------------
elif mode=="Final Decision":
    header("Final Decision","Current procurement actions and the latest decision changes generated by ProcuraX.","DECISION SUMMARY")

    # Final Decision is always populated from the active decision engine.
    # A live Data Hub run adds a before/after comparison, but the manager should
    # never see an empty page when an active recommendation set already exists.
    current = dec[["SKU","Material","Category","UOM","Action","Priority","Recommended Qty","Recommended Supplier","Days Cover","Forecast Daily Demand","Supplier Score","Confidence %","Potential Savings","Reason"]].copy()
    prev = st.session_state.get("live_previous_decisions")
    live_active = bool(st.session_state.get("live_active", False)) and prev is not None

    # Final Decision is a true unresolved manager queue. Approved purchase decisions remain
    # visible only during the 2-hour recall window; after that they move to Procurement Workspace.
    unresolved_rows=[]
    for _,r in current.iterrows():
        w=st.session_state.approval_workflow.get(r.SKU,{})
        status=w.get("status","")
        if status in {"Rejected by Manager","Recalled by Manager"} or not status:
            unresolved_rows.append(r)
        elif status.startswith("Approved — Forwarded"):
            approved_at=_parse_ts(w.get("approved_at"))
            if approved_at and datetime.now() < approved_at + timedelta(hours=APPROVAL_RECALL_HOURS):
                unresolved_rows.append(r)
    current_queue=pd.DataFrame(unresolved_rows) if unresolved_rows else current.iloc[0:0].copy()

    # Show the latest run impact before the manager queue so the comparison is immediately visible.
    if live_active and st.session_state.get("live_latest_decisions") is not None:
        merged_latest, changed_latest = build_analysis_change_set(
            st.session_state.get("live_previous_decisions"),
            st.session_state.get("live_latest_decisions"),
        )
        decision_changes=int((merged_latest["Decision Changed"]=="YES").sum())
        supplier_changes=int((merged_latest["Supplier Changed"]=="YES").sum())
        quantity_changes=int((merged_latest["Quantity Changed"]=="YES").sum())
        st.markdown('<div class="section-title">Latest analysis change summary</div>',unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Only outputs affected by the latest Data Hub run are shown here. The comparison is the immediately preceding live decision set versus the current run.</div>',unsafe_allow_html=True)
        a,b,c,d=st.columns(4)
        a.markdown(f'<div class="metric"><div class="label">Analysis Run</div><div class="value">#{st.session_state.live_change_run}</div><div class="delta">Latest live dataset</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="metric"><div class="label">Decisions Changed</div><div class="value">{decision_changes}</div><div class="delta">Action category changed</div></div>',unsafe_allow_html=True)
        c.markdown(f'<div class="metric"><div class="label">Supplier Changes</div><div class="value">{supplier_changes}</div><div class="delta">Recommended supplier changed</div></div>',unsafe_allow_html=True)
        d.markdown(f'<div class="metric"><div class="label">Quantity Changes</div><div class="value">{quantity_changes}</div><div class="delta">Recommended quantity changed</div></div>',unsafe_allow_html=True)
        if len(changed_latest):
            display_latest=changed_latest[["SKU","Material","Action_Before","Action_After","Recommended Qty_Before","Recommended Qty_After","Recommended Supplier_Before","Recommended Supplier_After","Days Cover_Before","Days Cover_After","Confidence %_Before","Confidence %_After","Changed Fields","Change Driver"]].copy()
            display_latest.columns=["SKU","Material","Decision Before","Decision After","Qty Before","Qty After","Supplier Before","Supplier After","Days Cover Before","Days Cover After","Confidence Before","Confidence After","Changed Fields","Change Driver"]
            st.dataframe(style_analysis_comparison(display_latest),use_container_width=True,hide_index=True)
        else:
            st.success("The latest Data Hub change did not alter any procurement action, supplier recommendation or recommended quantity.")

    st.markdown('<div class="section-title">Current procurement decisions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Manager review queue. Approved purchase decisions remain here only during the 2-hour Recall Approval window, then move automatically to Procurement Workspace.</div>', unsafe_allow_html=True)
    final_target=st.session_state.get("final_target_sku")
    if final_target and final_target in set(current_queue["SKU"]):
        target_final_row=current_queue[current_queue["SKU"]==final_target]
        st.success(f"Opened from AI Decisions: {target_final_row.iloc[0]['Material']} · {final_target}. Review the recommendation below and approve when ready.")
    final_family=st.selectbox("Component family",component_family_options,key="final_family")
    final_dec=current_queue.copy() if final_family=="All component families" else current_queue[current_queue["Category"]==final_family].copy()
    final_priority=st.selectbox("Priority view",["All priorities","Critical","Watch","Normal"],key="final_priority")
    if final_priority!="All priorities":
        final_dec=final_dec[final_dec["Priority"]==final_priority].copy()
    final_dec=final_dec.assign(_priority_rank=final_dec["Priority"].map(priority_rank)).sort_values(["_priority_rank","Confidence %"],ascending=[True,False]).drop(columns=["_priority_rank"])
    if final_target and final_target in set(final_dec["SKU"]):
        _target_final = final_dec[final_dec["SKU"]==final_target]
        final_dec = pd.concat([_target_final, final_dec[final_dec["SKU"]!=final_target]], ignore_index=True)
    if final_dec.empty:
        st.info("No unresolved decisions match the selected filters. Approved purchase decisions are shown in Procurement Workspace after the 2-hour recall window.")

    # Final Decision is the only place where the manager can approve, reject, or recall a procurement action.
    card_idx=0
    for priority_name, priority_title in [("Critical","Critical Decisions"),("Watch","Watch Decisions"),("Normal","Normal Decisions")]:
        priority_subset=final_dec[final_dec["Priority"]==priority_name]
        if priority_subset.empty:
            continue
        heading_pill="pill-red" if priority_name=="Critical" else ("pill-amber" if priority_name=="Watch" else "pill-green")
        st.markdown(f'<div class="section-title" style="margin-top:24px">{priority_title} <span class="pill {heading_pill}" style="margin-left:8px">{len(priority_subset)}</span></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{("Immediate manager review required before replenishment risk increases." if priority_name=="Critical" else "Review soon to protect continuity and avoid avoidable expediting." if priority_name=="Watch" else "Lower-priority recommendations that can be reviewed after urgent decisions.")}</div>',unsafe_allow_html=True)
        for _, r in priority_subset.iterrows():
            cls="critical" if r.Priority=="Critical" else ("watch" if r.Priority=="Watch" else "")
            pill="pill-red" if r.Priority=="Critical" else ("pill-amber" if r.Priority=="Watch" else "pill-green")
            st.markdown(f'<div class="ai-card {cls}"><span class="pill {pill}">{r.Action}</span><div class="ai-title">{r.Material} <span style="color:#7C8FA3;font-weight:600">· {r.SKU}</span></div><div class="ai-meta"><b>Supplier:</b> {r["Recommended Supplier"]} &nbsp; · &nbsp; <b>Qty:</b> {int(r["Recommended Qty"]):,} {r.get("UOM","unit")} &nbsp; · &nbsp; <b>Confidence:</b> {r["Confidence %"]:.0f}%</div><div class="ai-reason"><b>AI rationale:</b> {r.Reason}</div></div>',unsafe_allow_html=True)
            workflow=st.session_state.approval_workflow.get(r.SKU,{})
            is_purchase_action = r.Action in ["BUY NOW", "BUY SOON"]
            if not workflow.get("status"):
                ba,bb=st.columns([1,1],gap="medium")
                with ba:
                    approve_label = "✓ Approve & Forward" if is_purchase_action else "✓ Approve Decision"
                    if st.button(approve_label,key=f"final_approve_{r.SKU}_{card_idx}",type="primary",use_container_width=True):
                        st.session_state.decisions[r.SKU]="Approved"
                        if is_purchase_action:
                            st.session_state.approval_workflow[r.SKU]={"status":"Approved — Forwarded to Purchaser","approved_at":datetime.now().isoformat(),"purchaser":current_manager["purchaser"],"supplier":r["Recommended Supplier"],"qty":int(r["Recommended Qty"]),"recall_window_hours":APPROVAL_RECALL_HOURS}
                        else:
                            st.session_state.approval_workflow[r.SKU]={"status":f"Approved — {r.Action.title()} Procurement","purchaser":current_manager["purchaser"],"supplier":r["Recommended Supplier"],"qty":0}
                        persist_current_state()
                        st.rerun()
                with bb:
                    if st.button("Reject",key=f"final_reject_{r.SKU}_{card_idx}",use_container_width=True):
                        st.session_state.decisions[r.SKU]="Rejected"
                        st.session_state.approval_workflow[r.SKU]={"status":"Rejected by Manager","purchaser":current_manager["purchaser"],"supplier":r["Recommended Supplier"],"qty":0,"rejection_reason":""}
                        persist_current_state()
                        st.rerun()
            else:
                status = workflow.get("status", "")
                if is_purchase_action:
                    status_detail = f'<p><b>{status}</b><br>Purchaser: <b>{workflow.get("purchaser", current_manager["purchaser"])}</b></p>'
                else:
                    status_detail = f'<p><b>{status}</b></p>'
                st.markdown(f'<div class="insight workflow-status-card"><div class="insight-title">WORKFLOW STATUS</div>{status_detail}</div>',unsafe_allow_html=True)
                if status.startswith("Approved — Forwarded"):
                    approved_at=_parse_ts(workflow.get("approved_at"))
                    expires=(approved_at + timedelta(hours=APPROVAL_RECALL_HOURS)) if approved_at else datetime.now()
                    remaining=max(0,int((expires-datetime.now()).total_seconds()))
                    mins=remaining//60
                    secs=remaining%60
                    st.markdown(f'<div class="insight" style="margin-top:12px"><div class="insight-title">RECALL APPROVAL WINDOW</div><p>Decision approved and forwarded. Recall remains available for <b>{mins}m {secs:02d}s</b>. After this window, the decision moves to PO Processing.</p></div>',unsafe_allow_html=True)
                    if st.button("↩ Recall Approval", key=f"final_recall_{r.SKU}_{card_idx}", use_container_width=True):
                        st.session_state.decisions[r.SKU]="Pending"
                        st.session_state.approval_workflow[r.SKU]={"status":"Recalled by Manager","purchaser":workflow.get("purchaser",current_manager["purchaser"]),"supplier":workflow.get("supplier",r["Recommended Supplier"]),"qty":int(workflow.get("qty",0) or 0)}
                        persist_current_state()
                        st.rerun()
                elif workflow.get("status") == "Recalled by Manager":
                    st.markdown('<div class="workflow-message-spacer"></div>',unsafe_allow_html=True)
                    st.markdown('<div class="workflow-message">',unsafe_allow_html=True)
                    st.info("Purchase task recalled by the manager. Approve it again when ready.")
                    st.markdown('</div>',unsafe_allow_html=True)
                    ra, rb = st.columns([1, 1], gap="medium")
                    with ra:
                        if st.button("✓ Approve & Forward Again", key=f"final_reapprove_{r.SKU}_{card_idx}", type="primary", use_container_width=True):
                            st.session_state.decisions[r.SKU]="Approved"
                            st.session_state.approval_workflow[r.SKU]={"status":"Approved — Forwarded to Purchaser","approved_at":datetime.now().isoformat(),"purchaser":current_manager["purchaser"],"supplier":r["Recommended Supplier"],"qty":int(r["Recommended Qty"]),"recall_window_hours":APPROVAL_RECALL_HOURS}
                            persist_current_state()
                            st.rerun()
                    with rb:
                        if st.button("Reject", key=f"final_reject_recalled_{r.SKU}_{card_idx}", use_container_width=True):
                            st.session_state.decisions[r.SKU]="Rejected"
                            st.session_state.approval_workflow[r.SKU]={"status":"Rejected by Manager","purchaser":current_manager["purchaser"],"supplier":r["Recommended Supplier"],"qty":0,"rejection_reason":""}
                            persist_current_state()
                            st.rerun()
                elif workflow.get("status") == "Rejected by Manager":
                    st.markdown('<div class="workflow-rejection-box">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title" style="color:#B91C1C">REJECTION REVIEW</div>', unsafe_allow_html=True)
                    st.markdown("<div style='color:#52657B;font-size:.86rem;margin:4px 0 8px'>The recommendation is paused. Record the manager's reason before choosing the next action.</div>", unsafe_allow_html=True)
                    reason_options=[
                        "Quantity not appropriate",
                        "Supplier not preferred",
                        "Price not acceptable",
                        "Timing not appropriate",
                        "Need further review",
                        "Other / Manager note",
                    ]
                    saved_reason=workflow.get("rejection_reason", "")
                    default_index=reason_options.index(saved_reason) if saved_reason in reason_options else 0
                    selected_reason=st.selectbox("Rejection reason", reason_options, index=default_index, key=f"reject_reason_{r.SKU}_{card_idx}")
                    note=""
                    if selected_reason == "Other / Manager note":
                        note=st.text_input("Manager note", value=workflow.get("rejection_note", ""), key=f"reject_note_{r.SKU}_{card_idx}", placeholder="Briefly explain why the recommendation was rejected")
                    final_reason=note.strip() if selected_reason == "Other / Manager note" else selected_reason
                    st.session_state.approval_workflow[r.SKU]["rejection_reason"]=final_reason
                    if final_reason:
                        st.markdown(f'<div style="margin-top:8px;color:#7F1D1D;font-size:.82rem"><b>Reason recorded:</b> {final_reason}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    rr1, rr2 = st.columns([1, 1], gap="medium")
                    with rr1:
                        if st.button("↻ Send for Re-analysis", key=f"final_reanalyse_{r.SKU}_{card_idx}", use_container_width=True, disabled=not bool(final_reason)):
                            st.session_state.reanalysis_requested[r.SKU]={"reason":final_reason,"timestamp":datetime.now().isoformat()}
                            st.session_state.decisions[r.SKU]="Pending"
                            st.session_state.approval_workflow.pop(r.SKU, None)
                            st.session_state.selected_mode="AI Decisions"
                            persist_current_state()
                            st.rerun()
                    with rr2:
                        if st.button("↺ Reopen Decision", key=f"final_reopen_{r.SKU}_{card_idx}", use_container_width=True, disabled=not bool(final_reason)):
                            st.session_state.decisions[r.SKU]="Pending"
                            st.session_state.approval_workflow.pop(r.SKU, None)
                            persist_current_state()
                            st.rerun()
                elif status.startswith("Approved"):
                    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="workflow-task"><b>PROCUREMENT ACTION</b><br>{r.Action.title()} confirmed. No PO release is required at this stage.</div>',unsafe_allow_html=True)
                    # Delay / Monitor decisions do not create a purchaser task, so use a
                    # semantically correct reopen action rather than a purchaser callback.
                    if not is_purchase_action:
                        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                        if st.button("↩ Reopen Decision", key=f"final_reopen_approved_{r.SKU}_{card_idx}", use_container_width=True):
                            st.session_state.decisions[r.SKU]="Pending"
                            st.session_state.approval_workflow.pop(r.SKU, None)
                            persist_current_state()
                            st.rerun()
                st.markdown('<div class="final-decision-action-gap"></div>',unsafe_allow_html=True)
                card_idx += 1

    st.markdown('<div class="section-title">Decision History & Audit Trail</div>',unsafe_allow_html=True)
    if st.session_state.decision_history:
        hist_view=pd.DataFrame(st.session_state.decision_history)
        rename_map={"run":"Run","timestamp":"Timestamp","open_decisions":"Open decisions","critical":"Critical","savings":"Savings opportunity","forecast_model":"Forecast model"}
        hist_view=hist_view.rename(columns=rename_map)
        st.dataframe(hist_view.sort_values("Run",ascending=False),use_container_width=True,hide_index=True)
    else:
        st.caption("No analysis runs recorded yet.")

# -----------------------------
# Settings
# -----------------------------
elif mode=="Settings":
    header("Settings","Configure the ProcuraX decision engine, procurement policy and governance controls.","SYSTEM")
    cfg=st.session_state.ai_settings
    try:
        active_model,_= _resolve_forecast_model(hist,cfg)
        metrics = evaluate_forecast_models(hist,int(cfg.get("history_window",365)),bool(cfg.get("seasonality_enabled",True)),bool(cfg.get("trend_enabled",True))) if cfg.get("forecast_model")=="Auto Select" else None
    except RuntimeError as e:
        active_model = "Random Forest" if cfg.get("forecast_model") in ("XGBoost","Auto Select") else str(cfg.get("forecast_model","Random Forest"))
        metrics = None
        st.warning(str(e))

    st.markdown(f'<div class="panel"><div class="section-title" style="margin-top:0">AI Decision Engine</div><p style="color:#52657B;font-size:.82rem;margin-bottom:10px">Configure how ProcuraX forecasts demand, evaluates inventory risk and generates procurement recommendations. Changes are saved automatically for this manager.</p><span class="pill pill-teal">ACTIVE MODEL · {active_model}</span></div>',unsafe_allow_html=True)

    with st.expander("Demand forecasting", expanded=True):
        c1,c2,c3=st.columns(3)
        with c1:
            model_options = FORECAST_MODEL_OPTIONS if XGBRegressor is not None else ["Random Forest","Neural Network","Auto Select"]
            current_model = cfg.get("forecast_model","Random Forest")
            if current_model not in model_options:
                current_model = "Random Forest"
                cfg["forecast_model"] = current_model
            cfg["forecast_model"]=st.selectbox("Forecasting model",model_options,index=model_options.index(current_model),key="settings_forecast_model")
            if XGBRegressor is None:
                st.caption("XGBoost is currently unavailable in this Python environment. Use the ProcuraX launcher or install dependencies with the included installer to enable it.")
        with c2:
            cfg["forecast_horizon"]=st.selectbox("Forecast horizon (days)",[7,14,30,60,90],index=[7,14,30,60,90].index(int(cfg.get("forecast_horizon",30))),key="settings_forecast_horizon")
        with c3:
            cfg["history_window"]=st.selectbox("Historical demand window",[90,180,365,730],index=[90,180,365,730].index(int(cfg.get("history_window",365))),key="settings_history_window",format_func=lambda x:f"{x} days")
        c4,c5,c6=st.columns(3)
        with c4: cfg["seasonality_enabled"]=st.toggle("Seasonality detection",value=bool(cfg.get("seasonality_enabled",True)),key="settings_seasonality")
        with c5: cfg["trend_enabled"]=st.toggle("Trend detection",value=bool(cfg.get("trend_enabled",True)),key="settings_trend")
        with c6: cfg["confidence_threshold"]=st.selectbox("Minimum confidence",[60,65,70,75,80,85,90],index=[60,65,70,75,80,85,90].index(int(cfg.get("confidence_threshold",70))),key="settings_confidence",format_func=lambda x:f"{x}%")
        if cfg.get("forecast_model")=="Auto Select":
            if XGBRegressor is None:
                st.info("Auto Select is evaluating the forecasting models currently available in this environment. Install the ProcuraX dependencies to include XGBoost.")
            if metrics is not None and not metrics.empty:
                best=metrics.iloc[0]
                st.markdown(f'<div class="insight"><div class="insight-title">AUTO SELECT ACTIVE · {best["Model"]}</div><p>ProcuraX compares the available forecasting models on a time-based validation sample and selects the lowest-error model. Current validation winner: <b>{best["Model"]}</b> · MAE {best["MAE"]:.2f} · MAPE {best["MAPE"]:.1f}%.</p></div>',unsafe_allow_html=True)
                st.dataframe(metrics.style.format({"MAE":"{:.2f}","MAPE":"{:.1f}%"}),use_container_width=True,hide_index=True)
        else:
            st.caption(f"Active forecasting model: **{active_model}**. Random Forest, XGBoost and Neural Network are functional forecasting options; Auto Select evaluates them before choosing the lowest-validation-error model.")

    with st.expander("Inventory decision logic", expanded=False):
        c1,c2,c3=st.columns(3)
        with c1: cfg["safety_stock_policy"]=st.selectbox("Safety stock policy",["AI Calculated","Coverage Based"],index=["AI Calculated","Coverage Based"].index(cfg.get("safety_stock_policy","AI Calculated")),key="settings_safety_policy")
        with c2: cfg["safety_stock_days"]=st.selectbox("Safety stock coverage",[3,5,7,10,14],index=[3,5,7,10,14].index(int(cfg.get("safety_stock_days",7))),key="settings_safety_days",format_func=lambda x:f"{x} days")
        with c3: cfg["min_supplier_score"]=st.selectbox("Minimum supplier score",[60,65,70,75,80,85],index=[60,65,70,75,80,85].index(int(cfg.get("min_supplier_score",70))),key="settings_supplier_min",format_func=lambda x:f"{x} / 100")
        c4,c5=st.columns(2)
        with c4: cfg["incoming_po_consideration"]=st.toggle("Consider incoming purchase orders",value=bool(cfg.get("incoming_po_consideration",True)),key="settings_incoming_po")
        with c5: cfg["stockout_protection"]=st.toggle("Stockout protection",value=bool(cfg.get("stockout_protection",True)),key="settings_stockout")

    with st.expander("Procurement decision policy", expanded=False):
        c1,c2=st.columns(2)
        with c1: cfg["buy_now_days"]=st.selectbox("BUY NOW · coverage below",[1,2,3,4,5],index=[1,2,3,4,5].index(int(cfg.get("buy_now_days",3))),key="settings_buy_now",format_func=lambda x:f"{x} days")
        with c2: cfg["buy_soon_days"]=st.selectbox("BUY SOON · coverage below",[5,7,10,14,21],index=[5,7,10,14,21].index(int(cfg.get("buy_soon_days",7))),key="settings_buy_soon",format_func=lambda x:f"{x} days")
        st.caption("ProcuraX applies these thresholds together with safety stock, supplier lead time and inventory position. Changing policy settings affects the next analysis run.")

    with st.expander("Confidence & exception handling", expanded=False):
        c1,c2=st.columns(2)
        with c1: cfg["low_confidence_review"]=st.toggle("Flag low-confidence recommendations",value=bool(cfg.get("low_confidence_review",True)),key="settings_low_conf")
        with c2: cfg["data_anomaly_detection"]=st.toggle("Detect unusual input values",value=bool(cfg.get("data_anomaly_detection",True)),key="settings_anomaly")
        st.caption(f"Recommendations below {cfg.get('confidence_threshold',70)}% confidence can be treated as exceptions for manual review.")

    with st.expander("Supplier intelligence", expanded=False):
        st.markdown(f"**Current scoring:** On-time delivery {st.session_state.supplier_weights[0]}% · Quality {st.session_state.supplier_weights[1]}% · Lead time {st.session_state.supplier_weights[2]}% · Cost competitiveness {st.session_state.supplier_weights[3]}%")
        st.caption("Detailed supplier weights remain managed in Supplier Intelligence to avoid duplicate controls.")
        if st.button("Open Supplier Intelligence →",key="settings_open_supplier",use_container_width=False):
            st.session_state.selected_mode="Supplier Intelligence"; st.rerun()

    with st.expander("Approval & procurement workflow", expanded=False):
        c1,c2=st.columns(2)
        with c1: cfg["approval_recall_hours"]=st.selectbox("Approval recall window",[1,2,4,8],index=[1,2,4,8].index(int(cfg.get("approval_recall_hours",2))),key="settings_recall_window",format_func=lambda x:f"{x} hour" if x==1 else f"{x} hours")
        with c2: cfg["po_processing_sla_days"]=st.selectbox("PO processing SLA",[2,3,4,5,7],index=[2,3,4,5,7].index(int(cfg.get("po_processing_sla_days",4))),key="settings_po_sla",format_func=lambda x:f"{x} days")
        st.caption("Workflow timers are evaluated against the persisted approval timestamps. Existing approved tasks continue under their recorded timestamps.")

    with st.expander("Decision governance", expanded=False):
        st.info("Recommendations are advisory. Procurement managers remain responsible for approval, modification or rejection of a recommendation.")
        c1,c2,c3=st.columns(3)
        c1.markdown('<div class="metric"><div class="label">AI recommendations</div><div class="value" style="font-size:1.25rem">Advisory</div><div class="delta">Human decision required</div></div>',unsafe_allow_html=True)
        c2.markdown('<div class="metric"><div class="label">Explainability</div><div class="value" style="font-size:1.25rem">Enabled</div><div class="delta">Reason shown with decisions</div></div>',unsafe_allow_html=True)
        c3.markdown('<div class="metric"><div class="label">Audit trail</div><div class="value" style="font-size:1.25rem">Enabled</div><div class="delta">Actions persisted locally</div></div>',unsafe_allow_html=True)

    with st.expander("Data & system management", expanded=False):
        db_path=database.DB_PATH
        size_kb=(db_path.stat().st_size/1024) if db_path.exists() else 0
        mtime=datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%d %b %Y · %H:%M") if db_path.exists() else "Not created yet"
        c1,c2,c3=st.columns(3)
        c1.markdown(f'<div class="metric"><div class="label">Database status</div><div class="value" style="font-size:1.15rem">● Active</div><div class="delta">SQLite · {size_kb:.0f} KB</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="metric"><div class="label">Last database update</div><div class="value" style="font-size:1.05rem">{mtime}</div><div class="delta">Persistent manager state</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="metric"><div class="label">Current analysis runs</div><div class="value">{int(st.session_state.live_analysis_version)}</div><div class="delta">Stored for this manager</div></div>',unsafe_allow_html=True)
        st.caption("ProcuraX stores operational state locally in procurax.db. No separate database server is required.")
        e1,e2=st.columns(2)
        with e1:
            if st.session_state.decision_history:
                export_df=pd.DataFrame(st.session_state.decision_history)
                st.download_button("Export Decision History",export_df.to_csv(index=False).encode("utf-8"),file_name=f"procurax_{st.session_state.username}_decision_history.csv",mime="text/csv",use_container_width=True,key="settings_export_history")
            else:
                st.button("Export Decision History",disabled=True,use_container_width=True,key="settings_export_history_disabled")
        with e2:
            if st.button("Reset Demo Data",use_container_width=True,key="settings_reset_demo"):
                st.session_state.live_active=False
                st.session_state.live_materials=(demo_hist[demo_hist["Procurement_Category"]==current_manager["category"]].sort_values("Date").groupby("SKU").tail(1)[["SKU","Material","Category","Procurement_Category","UOM","Consumption","Current_Stock","Unit_Price"]].copy()).reset_index(drop=True)
                st.session_state.live_suppliers=demo_sup[demo_sup["Procurement_Category"]==current_manager["category"]].copy().reset_index(drop=True)
                st.session_state.live_overrides={}
                st.session_state.live_previous_decisions=None
                st.session_state.live_latest_decisions=None
                st.session_state.live_change_run=None
                st.session_state.live_analysis_version=0
                st.session_state.approval_workflow={}
                st.session_state.inventory_receipts={}
                st.session_state.decisions={}
                st.session_state.reanalysis_requested={}
                st.session_state.decision_history=[]
                persist_current_state()
                st.success("Demo data and persisted operational state have been reset.")
                st.rerun()

    with st.expander("External intelligence · Future integrations", expanded=False):
        st.caption("These connectors are intentionally shown as future-ready controls. They are not active in the current MVP.")
        c1,c2,c3=st.columns(3)
        c1.markdown('<div class="panel"><b>Commodity prices</b><br><span style="color:#64748B;font-size:.78rem">○ Future integration</span></div>',unsafe_allow_html=True)
        c2.markdown('<div class="panel"><b>Freight intelligence</b><br><span style="color:#64748B;font-size:.78rem">○ Future integration</span></div>',unsafe_allow_html=True)
        c3.markdown('<div class="panel"><b>FX / market signals</b><br><span style="color:#64748B;font-size:.78rem">○ Future integration</span></div>',unsafe_allow_html=True)

    # Persist configuration changes on every Settings rerun so the selected policy is retained.
    persist_current_state()

elif mode=="User Management":
    header("User Management", "Manage ProcuraX users, procurement categories and access without touching the application code.", eyebrow="ADMINISTRATION")
    users = database.list_users()
    active_count = sum(1 for u in users if bool(u.get("enabled", 1)))
    manager_count = sum(1 for u in users if u.get("role") == "manager")
    category_count = len({u.get("category") for u in users if u.get("role") == "manager"})
    st.markdown(f"""<div class="admin-summary-grid"><div class="admin-stat"><div class="admin-stat-label">ACTIVE USERS</div><div class="admin-stat-value">{active_count}</div><div class="admin-stat-sub">Accounts currently enabled</div></div><div class="admin-stat"><div class="admin-stat-label">PROCUREMENT MANAGERS</div><div class="admin-stat-value">{manager_count}</div><div class="admin-stat-sub">Manager profiles configured</div></div><div class="admin-stat"><div class="admin-stat-label">CATEGORIES IN USE</div><div class="admin-stat-value">{category_count}</div><div class="admin-stat-sub">Assigned procurement domains</div></div></div>""", unsafe_allow_html=True)
    rows_html=[]
    for u in users:
        status_cls="active" if bool(u.get("enabled",1)) else "disabled"; status_txt="Active" if bool(u.get("enabled",1)) else "Disabled"; role="Master Admin" if u.get("role")=="admin" else "Manager"
        rows_html.append(f'<tr><td><div class="admin-user-name">{html.escape(str(u.get("display_name") or "—"))}</div><div class="admin-username">@{html.escape(str(u.get("username") or "—"))}</div></td><td><span class="admin-role">{html.escape(role)}</span></td><td>{html.escape(str(u.get("category") or "—"))}</td><td>{html.escape(str(u.get("job_role") or "—"))}</td><td><span class="admin-status {status_cls}">● {status_txt}</span></td></tr>')
    st.markdown(f"""<div class="admin-section"><div class="admin-section-head"><div><div class="admin-section-title">Users & access</div><div class="admin-section-sub">A quick view of every account and its assigned procurement scope.</div></div></div><table class="admin-table"><thead><tr><th>User</th><th>Role</th><th>Procurement category</th><th>Job role</th><th>Status</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>""", unsafe_allow_html=True)
    st.markdown('''<div class="admin-section"><div class="admin-section-head"><div><div class="admin-section-title">Add user / manager</div><div class="admin-section-sub">Create a new account and assign its procurement domain.</div></div></div>''', unsafe_allow_html=True)
    with st.form("admin_add_user_form", clear_on_submit=True):
        c1,c2,c3=st.columns(3)
        with c1: new_name=st.text_input("Display name",placeholder="e.g. Rahul Shah")
        with c2: new_username=st.text_input("Username",placeholder="e.g. rahul")
        with c3: new_password=st.text_input("Temporary password",type="password",placeholder="Set an initial password")
        c4,c5,c6=st.columns(3)
        with c4: category_choice=st.selectbox("Procurement category",AVAILABLE_CATEGORIES+["Custom / dataset not configured"],key="admin_add_category")
        with c5: new_job_role=st.text_input("Job role",value="Procurement Manager")
        with c6: new_purchaser=st.text_input("Assigned purchaser",value="—")
        custom_category=st.text_input("Custom category name",placeholder="e.g. Television Components") if category_choice=="Custom / dataset not configured" else category_choice
        submitted_add=st.form_submit_button("Add User →",type="primary")
    st.markdown('<div class="admin-note"><b>Access rule:</b> managers only see data belonging to their assigned procurement category. A category without a packaged dataset will open safely as <b>Dataset not configured</b> rather than causing a dashboard error.</div></div>',unsafe_allow_html=True)
    if submitted_add:
        try:
            final_category=custom_category.strip()
            if not final_category: raise ValueError("Category is required.")
            database.create_user(new_username,new_name,new_password,final_category,"manager",new_job_role,new_purchaser)
            st.success(f"User {new_username.strip().lower()} created successfully."); st.rerun()
        except Exception as exc: st.error(str(exc))
    st.markdown('''<div class="admin-section"><div class="admin-section-head"><div><div class="admin-section-title">Edit access</div><div class="admin-section-sub">Update a user's role, procurement scope, purchaser assignment or active status.</div></div></div>''',unsafe_allow_html=True)
    usernames=[u["username"] for u in users]
    if usernames:
        selected_user=st.selectbox("Select user to manage",usernames,key="admin_selected_user"); selected=database.get_user(selected_user)
        if selected:
            with st.form("admin_edit_user_form"):
                e1,e2,e3=st.columns(3)
                with e1: edit_name=st.text_input("Display name",value=selected["display_name"],key="admin_edit_name")
                with e2: edit_role=st.selectbox("Role",["manager","admin"],index=0 if selected["role"]=="manager" else 1,key="admin_edit_role")
                with e3: edit_enabled=st.checkbox("Account active",value=bool(selected["enabled"]),key="admin_edit_enabled")
                f1,f2,f3=st.columns(3); category_options=AVAILABLE_CATEGORIES if selected["category"] in AVAILABLE_CATEGORIES else AVAILABLE_CATEGORIES+[selected["category"]]
                with f1: edit_category=st.selectbox("Procurement category",category_options,index=category_options.index(selected["category"]),key="admin_edit_category")
                with f2: edit_job=st.text_input("Job role",value=selected.get("job_role") or "Procurement Manager",key="admin_edit_job")
                with f3: edit_purchaser=st.text_input("Assigned purchaser",value=selected.get("purchaser") or "—",key="admin_edit_purchaser")
                new_pw=st.text_input("Reset password (optional)",type="password",placeholder="Leave blank to keep current password",key="admin_edit_password")
                save_edit=st.form_submit_button("Save Access Changes →",type="primary")
            if save_edit:
                if selected_user==st.session_state.username and (not edit_enabled or edit_role!="admin"): st.error("You cannot disable or demote the currently signed-in Master Admin.")
                else:
                    try:
                        database.update_user(selected_user,edit_name,edit_category,edit_role,edit_job,edit_purchaser,edit_enabled,new_pw or None); st.success("Access settings updated successfully."); st.rerun()
                    except Exception as exc: st.error(str(exc))
    st.markdown('<div class="admin-danger-note"><b>Security note:</b> passwords are never displayed in the admin console. Resetting a password replaces the stored password hash; existing sessions are not exposed.</div></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">ProcuraX MVP · AI procurement decision support · Predictive modelling + supplier scoring + business rules · Human approval required</div>',unsafe_allow_html=True)
