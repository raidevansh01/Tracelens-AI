"""Streamlit visual language with native, accessible widgets."""
import streamlit as st

STYLE = r'''
<style>
/* TraceLens / Slate and sea-glass workspace */
:root {--tl-ink:#17273e; --tl-muted:#63758a; --tl-line:#dce5ef; --tl-teal:#087f83; --tl-bg:#f6f9fc}
.stApp {background:var(--tl-bg);color:var(--tl-ink);}
[data-testid="stSidebar"]{background:#102438!important;border-right:0;}
[data-testid="stSidebar"] * {color:#edf5ff;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {color:#ecf4fa;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:8px;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {background:#19334a;border:1px solid #284458;border-radius:10px;padding:9px 12px;transition:background .15s;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked){background:#17606b;border-color:#4cc7c3;}
.block-container {max-width:1360px;padding:2rem 2.3rem 5rem;}
.stApp h1,.stApp h2,.stApp h3{font-family:inherit;letter-spacing:-.025em;color:#182d43;}
.stApp h1{font-size:2rem;line-height:1.15;}
.stApp h2{font-size:1.42rem;}
.stApp h3{font-size:1.1rem;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--tl-line);border-radius:14px;padding:14px 18px;min-height:117px;}
[data-testid="stMetricLabel"]{color:var(--tl-muted);}
[data-testid="stMetricValue"]{color:var(--tl-ink);font-weight:750;font-size:1.85rem;}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--tl-line)!important;border-radius:14px!important;background:#fff;}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#087f83!important;border-color:#087f83!important;color:white!important;font-weight:650!important;}
.stButton button[kind="primary"]:hover,.stDownloadButton button[kind="primary"]:hover{background:#065e63!important;}
.stButton button:focus-visible, .stDownloadButton button:focus-visible{outline:3px solid #9ed6d6!important;outline-offset:2px;}
.tl-eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;color:#007c7d;margin:0 0 10px;}
.tl-hero{border:1px solid #d7e5ec;background:linear-gradient(115deg,#ffffff 0%,#edf7fa 100%);border-radius:20px;padding:26px 30px;margin-bottom:18px;}
.tl-hero h1{font-size:2.12rem!important;margin:0 0 12px;color:#17354a;}
.tl-hero p{font-size:1rem;color:#52687b;max-width:830px;margin:0;line-height:1.65;}
.tl-chip{display:inline-block;padding:5px 11px;border-radius:999px;background:#e5f3f1;color:#056b66;font-size:.78rem;font-weight:700;margin-right:6px;margin-bottom:5px;}
.tl-section{margin-top:1.4rem;margin-bottom:.35rem;}
.tl-section .tllabel{font-size:.69rem;letter-spacing:.14em;text-transform:uppercase;color:#087f83;font-weight:800;}
.tl-section h3{margin:.26rem 0 .2rem;font-size:1.25rem;}
.tl-section p{color:#657689;font-size:.9rem;margin:.15rem 0 0;}
.tl-compare{border:1px solid #dde6ed;border-radius:12px;background:#fbfdff;padding:18px 20px;line-height:1.77;overflow-wrap:anywhere;white-space:pre-wrap;max-height:420px;overflow-y:auto;font-size:.94rem;}
.tl-compare mark{background:#ffe7a5;color:#242b33;padding:1px 2px;border-radius:3px;}
.tl-excerpt{background:#f5f9fb;padding:16px 18px;border:1px solid #e4edf3;border-radius:12px;white-space:pre-wrap;max-height:300px;overflow-y:auto;overflow-wrap:anywhere;line-height:1.7;}
.tl-brand{border:1px solid #3c6171;background:#183c50;border-radius:14px;padding:16px;margin-bottom:16px;color:#fff;}
.tl-brand .name{font-size:1.3rem;font-weight:800;color:#fff;letter-spacing:-.04em;}
.tl-brand small{display:block;color:#b4d7dc;margin-top:3px;font-size:.76rem;}
.tl-note{font-size:.85rem;color:#607789;line-height:1.6;margin-top:7px;}
.tl-status{display:inline-block;color:#1d7772;background:#e3f5ef;border-radius:6px;padding:3px 9px;font-weight:750;font-size:.78rem;}
.tl-foot{border-top:1px solid #dbe5eb;margin-top:2.2rem;padding-top:1rem;color:#728596;font-size:.79rem;}
@media (max-width:760px){.block-container{padding:1rem .9rem 3rem}.tl-hero{padding:19px 17px}.tl-hero h1{font-size:1.62rem!important}}
</style>
'''


def init_theme():
    st.markdown(STYLE, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, description: str):
    import html
    st.markdown(f'<div class="tl-hero"><div class="tl-eyebrow">{html.escape(eyebrow)}</div>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>',
                unsafe_allow_html=True)


def section(eyebrow: str, title: str, description: str = ''):
    import html
    st.markdown(f'<div class="tl-section"><div class="tllabel">{html.escape(eyebrow)}</div>'
                f'<h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>',
                unsafe_allow_html=True)
