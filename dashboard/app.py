import io, os, base64, requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="NeuroAI · Brain Tumor Staging",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

LABELS_4 = {
    0: ("No Tumor",           "#059669", "🟢", "Negative Control",       "Low",      "#D1FAE5", "#6EE7B7"),
    1: ("Meningioma",         "#0284C7", "🔵", "WHO Grade I — Benign",   "Low",      "#E0F2FE", "#7DD3FC"),
    2: ("Pituitary Tumor",    "#D97706", "🟡", "Local Extension",        "Moderate", "#FEF3C7", "#FCD34D"),
    3: ("Glioma III/IV",      "#DC2626", "🔴", "Malignant — Aggressive", "High",     "#FEE2E2", "#FCA5A5"),
}
LABELS_5 = {
    0: ("No Tumor",           "#059669", "🟢", "Negative Control",       "Low",      "#D1FAE5", "#6EE7B7"),
    1: ("Meningioma",         "#0284C7", "🔵", "WHO Grade I — Benign",   "Low",      "#E0F2FE", "#7DD3FC"),
    2: ("Pituitary Tumor",    "#D97706", "🟡", "Local Extension",        "Moderate", "#FEF3C7", "#FCD34D"),
    3: ("Anaplastic Glioma",  "#EA580C", "🟠", "WHO Grade III",          "High",     "#FFEDD5", "#FED7AA"),
    4: ("Glioblastoma GBM",   "#DC2626", "🔴", "WHO Grade IV",           "Critical", "#FEE2E2", "#FCA5A5"),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F1F5F9 !important;
    color: #1E293B !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 3rem !important; max-width: 1400px !important; }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 4px 6px;
    gap: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 0.855rem;
    font-weight: 500;
    color: #64748B;
    padding: 0.45rem 1.1rem;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #1E40AF !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.3) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.42) !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    background: #FAFBFC !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }

/* Selectbox */
[data-baseweb="select"] { border-radius: 8px !important; }

/* Dataframe */
[data-testid="stDataFrame"] iframe { border-radius: 10px !important; }

/* Progress */
.stProgress > div > div > div > div { background-color: #2563EB !important; }

/* Input */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    font-size: 0.85rem !important;
}

hr { border: none; border-top: 1px solid #E2E8F0 !important; margin: 0.8rem 0 !important; }

/* Spinner text */
.stSpinner > div { color: #2563EB !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=20)
def api_health(url):
    try:
        r = requests.get(f"{url}/", timeout=3)
        return r.json() if r.ok else None
    except:
        return None

def get_label(sid, n=4):
    return (LABELS_4 if n <= 4 else LABELS_5).get(sid,
        ("Unknown","#64748B","⚪","Unknown","Unknown","#F1F5F9","#CBD5E1"))

def kpi_html(value, label, sub, vcolor, bg="#FFFFFF"):
    return f"""<div style="background:{bg};border:1px solid #E2E8F0;border-radius:12px;
padding:1rem 1.1rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
<div style="font-size:1.55rem;font-weight:800;color:{vcolor};letter-spacing:-0.5px;line-height:1.1;">{value}</div>
<div style="font-size:0.68rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.6px;margin-top:3px;">{label}</div>
<div style="font-size:0.75rem;color:{vcolor};margin-top:2px;font-weight:500;opacity:0.85;">{sub}</div>
</div>"""

def clinical_report_txt(res, fname, ts):
    sid   = res["stage_id"]
    n     = res.get("num_classes", 4)
    lbl   = get_label(sid, n)
    lmap  = LABELS_4 if n <= 4 else LABELS_5
    probs = res["probabilities"]
    lines = [
           *60, "  NEUROAI — CLINICAL DIAGNOSTIC REPORT", "="*60,
        f"Date       : {ts}", f"File       : {fname}",
        f"Model      : EfficientNetB0 TL · 4-Class WHO Staging", "",
           *60, "  RESULT", "-"*60,
        f"Stage      : {lbl[0]}",  f"WHO Grade  : {lbl[3]}",
        f"Risk       : {lbl[4]}",  f"Confidence : {res['confidence']*100:.1f}%",
        f"Anomaly    : {res['anomaly_score']:.6f}",
        f"Review     : {'YES — '+res['review_reason'] if res['requires_review'] else 'NO'}", "",
           *60, "  CLASS PROBABILITIES", "-"*60,
    ]
    for i, p in enumerate(probs):
        nm = lmap.get(i,(str(i),))[0]
        b  = "█"*int(p*30) + "░"*(30-int(p*30))
        lines.append(f"  {nm:<26} {b} {p*100:5.1f}%")
    lines += ["", "-"*60,"  CLINICAL NOTE","-"*60, res.get("clinical_note","N/A"), "",
           *60,"  DISCLAIMER","="*60,
                                                                 ,
                                                                 ,"="*60]
    return "\n".join(lines)

st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A8A 0%,#2563EB 60%,#1D4ED8 100%);
border-radius:16px;padding:1.6rem 2.2rem;margin-bottom:1.4rem;
box-shadow:0 8px 30px rgba(37,99,235,0.22);display:flex;
align-items:center;justify-content:space-between;">
  <div>
    <div style="font-size:1.7rem;font-weight:800;color:#FFF;letter-spacing:-0.5px;">🧠 NeuroAI</div>
    <div style="font-size:0.84rem;color:#BFDBFE;margin-top:0.25rem;">
      Brain Tumor Staging · EfficientNetB0 Transfer Learning · Clinical Decision Support
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:0.8rem;">
      <span style="background:rgba(255,255,255,0.18);color:#FFF;border-radius:20px;padding:3px 11px;font-size:0.72rem;font-weight:600;border:1px solid rgba(255,255,255,0.3);">✓ Recall 97.8%</span>
      <span style="background:rgba(255,255,255,0.18);color:#FFF;border-radius:20px;padding:3px 11px;font-size:0.72rem;font-weight:600;border:1px solid rgba(255,255,255,0.3);">AUC 0.9863</span>
      <span style="background:rgba(255,255,255,0.18);color:#FFF;border-radius:20px;padding:3px 11px;font-size:0.72rem;font-weight:600;border:1px solid rgba(255,255,255,0.3);">Grad-CAM XAI</span>
      <span style="background:rgba(255,255,255,0.18);color:#FFF;border-radius:20px;padding:3px 11px;font-size:0.72rem;font-weight:600;border:1px solid rgba(255,255,255,0.3);">4-Class WHO</span>
      <span style="background:rgba(239,68,68,0.35);color:#FEE2E2;border-radius:20px;padding:3px 11px;font-size:0.72rem;font-weight:600;border:1px solid rgba(239,68,68,0.5);">Research Only</span>
    </div>
  </div>
  <div style="font-size:3.5rem;opacity:0.25;line-height:1;user-select:none;">🧬</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1E293B;margin:0 0 0.8rem;">⚙️ Configuration</p>', unsafe_allow_html=True)
    api_url  = st.text_input("", value=API_URL, placeholder="http://localhost:8000")
    api_info = api_health(api_url)

    if api_info:
        mname = os.path.basename(api_info.get("model_path") or "")
        ae    = api_info.get("autoencoder_loaded", False)
        st.markdown(f"""<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 13px;margin:6px 0 12px;">
<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
  <div style="width:8px;height:8px;border-radius:50%;background:#22C55E;flex-shrink:0;
  box-shadow:0 0 0 3px rgba(34,197,94,0.2);"></div>
  <b style="color:#15803D;font-size:0.85rem;">API Online</b>
</div>
<div style="font-size:0.74rem;color:#166534;line-height:1.6;">
  📦 {mname}<br>{api_info.get('num_classes','?')} classes · {api_info.get('img_size','?')}<br>
  {'✓' if ae else '✗'} Autoencoder {'loaded' if ae else 'missing'}
</div></div>""", unsafe_allow_html=True)
        num_cls_g = api_info.get("num_classes", 4)
    else:
        st.markdown("""<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:10px 13px;margin:6px 0 12px;">
<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
  <div style="width:8px;height:8px;border-radius:50%;background:#EF4444;flex-shrink:0;"></div>
  <b style="color:#B91C1C;font-size:0.85rem;">API Offline</b>
</div>
<div style="font-size:0.73rem;color:#991B1B;">Start: <code>uvicorn api.main:app --port 8000</code></div>
</div>""", unsafe_allow_html=True)
        num_cls_g = 4

    st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin:0 0 6px;">Stage Legend</p>', unsafe_allow_html=True)
    lsmap = LABELS_4 if num_cls_g <= 4 else LABELS_5
    for sid, (nm, col, ic, grade, risk, bg, acc) in lsmap.items():
        st.markdown(f"""<div style="background:{bg};border-left:3px solid {col};border-radius:8px;
padding:7px 11px;margin:4px 0;">
<div style="font-size:0.82rem;font-weight:600;color:{col};">{ic} {nm}</div>
<div style="font-size:0.71rem;color:#64748B;">{grade} · {risk} risk</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    conf_thr = st.slider("Confidence threshold", 0.50, 0.95, 0.70, 0.05)
    st.markdown("""<div style="text-align:center;padding:10px;background:#F8FAFC;border:1px solid #E2E8F0;
border-radius:8px;margin-top:1rem;font-size:0.68rem;color:#94A3B8;line-height:1.6;">
NeuroAI v4.0 · 2026<br>Research purposes only</div>""", unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.markdown(kpi_html("97.8%","Recall @ 0.70","✓ Target achieved","#15803D","#F0FDF4"), unsafe_allow_html=True)
k2.markdown(kpi_html("92.8%","Test Accuracy","1,600 images","#1D4ED8","#EFF6FF"),     unsafe_allow_html=True)
k3.markdown(kpi_html("0.9863","ROC-AUC","Excellent","#7C3AED","#F5F3FF"),             unsafe_allow_html=True)
k4.markdown(kpi_html("0.9268","F2-Score","Weighted recall","#D97706","#FFFBEB"),         unsafe_allow_html=True)
k5.markdown(kpi_html("< 200ms","Inference","CPU · no GPU","#0F766E","#F0FDFA"),       unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
                   , "📦  Batch", "📈  Performance", "📄  Reports", "ℹ️  About"
])

CLRS = ["#059669","#0284C7","#D97706","#DC2626","#7C3AED"]

with tab1:
    L, R = st.columns([1, 2], gap="large")

    with L:
        st.markdown('<p style="font-weight:700;color:#1E293B;margin-bottom:0.5rem;">📂 Upload MRI Scan</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
            arr = np.array(img)
            st.markdown(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
padding:10px 14px;margin-top:6px;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;">Image Info</div>
<table style="width:100%;font-size:0.78rem;border-collapse:collapse;">
<tr><td style="color:#94A3B8;padding:2px 0;">File</td><td style="color:#1E293B;font-weight:500;text-align:right;">{uploaded.name[:24]}</td></tr>
<tr><td style="color:#94A3B8;padding:2px 0;">Dimensions</td><td style="color:#1E293B;font-weight:500;text-align:right;">{img.size[0]} × {img.size[1]} px</td></tr>
<tr><td style="color:#94A3B8;padding:2px 0;">Mean / Std</td><td style="color:#1E293B;font-weight:500;text-align:right;">{arr.mean():.1f} / {arr.std():.1f}</td></tr>
</table></div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            analyze = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)
        else:
            st.markdown("""<div style="text-align:center;padding:3.5rem 1.5rem;background:#FFFFFF;
border:2px dashed #CBD5E1;border-radius:14px;">
<div style="font-size:2.5rem;">🧠</div>
<div style="font-size:0.9rem;font-weight:600;color:#475569;margin-top:0.5rem;">Drop your MRI scan here</div>
<div style="font-size:0.75rem;color:#94A3B8;margin-top:0.3rem;">JPEG · PNG</div>
</div>""", unsafe_allow_html=True)
            analyze = False

    with R:
        st.markdown('<p style="font-weight:700;color:#1E293B;margin-bottom:0.5rem;">🩺 Diagnostic Results</p>', unsafe_allow_html=True)

        if uploaded and analyze:
            with st.spinner("Running EfficientNetB0 + Grad-CAM…"):
                try:
                    uploaded.seek(0)
                    resp = requests.post(f"{api_url}/predict",
                        files={"file":(uploaded.name, uploaded.read(), uploaded.type)}, timeout=120)

                    if resp.status_code == 200:
                        res      = resp.json()
                        sid      = res["stage_id"]
                        conf     = res["confidence"]
                        probs    = res["probabilities"]
                        gcb64    = res["gradcam_base64"]
                        anom     = res["anomaly_score"]
                        rev      = res["requires_review"]
                        reason   = res["review_reason"]
                        note     = res["clinical_note"]
                        nclasses = res.get("num_classes", len(probs))
                        nm, col, ic, grade, risk, bg, acc = get_label(sid, nclasses)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        st.markdown(f"""<div style="background:{bg};border:1px solid {acc};
border-left:5px solid {col};border-radius:14px;padding:1.3rem 1.7rem;margin-bottom:0.8rem;
box-shadow:0 2px 12px rgba(0,0,0,0.06);">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:{col};">{ic} {nm}</div>
    <div style="font-size:0.82rem;color:#64748B;margin-top:3px;">{grade}</div>
    <div style="font-size:0.82rem;color:#475569;margin-top:7px;line-height:1.55;">{note}</div>
  </div>
  <div style="text-align:right;flex-shrink:0;margin-left:1rem;">
    <div style="font-size:0.65rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;">Risk</div>
    <div style="font-size:1rem;font-weight:700;color:{col};margin-top:2px;">{risk}</div>
  </div>
</div></div>""", unsafe_allow_html=True)

                        m1,m2,m3,m4 = st.columns(4)
                        cc = "#15803D" if conf >= conf_thr else "#DC2626"
                        ac = "#DC2626" if anom > 0.05 else "#15803D"
                        rc = "#DC2626" if rev else "#15803D"
                        m1.markdown(kpi_html(f"{conf*100:.1f}%","Confidence","",cc), unsafe_allow_html=True)
                        m2.markdown(kpi_html(f"Stage {sid}","Predicted","",col), unsafe_allow_html=True)
                        m3.markdown(kpi_html(f"{anom:.5f}","Anomaly MSE","",ac), unsafe_allow_html=True)
                        m4.markdown(kpi_html("⚠️ YES" if rev else "✅ NO","Review","",rc), unsafe_allow_html=True)

                        if rev:
                            bc = "#DC2626" if risk in ("High","Critical") else "#D97706"
                            bb = "#FEF2F2" if risk in ("High","Critical") else "#FFFBEB"
                            ia = "🚨 URGENT REVIEW" if risk in ("High","Critical") else "⚠️ Review Recommended"
                            st.markdown(f"""<div style="background:{bb};border:1px solid {bc};
border-left:4px solid {bc};border-radius:10px;padding:0.8rem 1.1rem;margin:0.6rem 0;">
<b style="color:{bc};">{ia}</b>
<span style="font-size:0.8rem;color:#64748B;margin-left:8px;">— {reason}</span>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<div style="background:#F0FDF4;border:1px solid #86EFAC;
border-left:4px solid #22C55E;border-radius:10px;padding:0.8rem 1.1rem;margin:0.6rem 0;">
<b style="color:#15803D;">✅ Confident Diagnosis</b>
<span style="font-size:0.8rem;color:#64748B;margin-left:8px;">
— Confidence {conf*100:.1f}% ≥ threshold {conf_thr*100:.0f}%</span>
</div>""", unsafe_allow_html=True)

                        st.markdown('<p style="font-size:0.85rem;font-weight:700;color:#1E293B;margin:0.8rem 0 0.4rem;">📊 Class Probabilities</p>', unsafe_allow_html=True)
                        lmd = LABELS_4 if nclasses <= 4 else LABELS_5
                        bars = ""
                        for i, p in enumerate(probs):
                            ln, lc, li, *_ = lmd.get(i,(str(i),"#888","⚪"))
                            w    = p * 100
                            bold = "font-weight:700" if i==sid else "font-weight:400"
                            bg_r = lmd.get(i,(None,None,None,None,None,"#FFFFFF"))[5] if i==sid else "#FFFFFF"
                            bars += f"""<div style="display:flex;align-items:center;gap:9px;
padding:5px 10px;border-radius:7px;background:{bg_r};margin:3px 0;">
<span style="font-size:0.78rem;min-width:145px;color:#1E293B;{bold};">{li} {ln}</span>
<div style="flex:1;height:7px;background:#F1F5F9;border-radius:4px;overflow:hidden;">
  <div style="height:100%;width:{w:.1f}%;background:{CLRS[i%5]};border-radius:4px;"></div>
</div>
<span style="min-width:42px;text-align:right;font-size:0.8rem;color:#1E293B;{bold};">{w:.1f}%</span>
</div>"""
                        st.markdown(f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:8px 6px;">{bars}</div>', unsafe_allow_html=True)

                        if gcb64:
                            st.markdown('<p style="font-size:0.85rem;font-weight:700;color:#1E293B;margin:0.8rem 0 0.4rem;">🔍 Grad-CAM Activation Map</p>', unsafe_allow_html=True)
                            gc_img = Image.open(io.BytesIO(base64.b64decode(gcb64)))
                            gc1, gc2 = st.columns(2)
                            gc1.image(img,    caption="Original MRI",    use_column_width=True)
                            gc2.image(gc_img, caption="Grad-CAM Overlay", use_column_width=True)
                            st.markdown("""<div style="background:#EFF6FF;border:1px solid #BFDBFE;
border-radius:8px;padding:8px 12px;font-size:0.76rem;color:#1D4ED8;margin-top:4px;">
🔬 <b>Grad-CAM:</b> Warm colors (red/yellow) = highest neural activation — regions that drove this decision.
</div>""", unsafe_allow_html=True)

                        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                        rpt = clinical_report_txt(res, uploaded.name, ts)
                        st.download_button("📄 Download Clinical Report", data=rpt.encode("utf-8"),
                            file_name=f"neuroai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain", use_container_width=True)

                    else:
                        st.markdown(f'<div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:0.9rem 1.2rem;color:#B91C1C;">❌ API Error {resp.status_code}: {resp.text[:300]}</div>', unsafe_allow_html=True)

                except requests.exceptions.ConnectionError:
                    st.markdown('<div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:0.9rem 1.2rem;color:#B91C1C;">🔌 Cannot reach API — <code>uvicorn api.main:app --port 8000</code></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))

        elif not uploaded:
            st.markdown("""<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:1.4rem 1.8rem;">
<p style="font-weight:700;color:#1D4ED8;font-size:0.92rem;margin:0 0 0.7rem;">📖 How to use NeuroAI</p>
<ol style="color:#3B82F6;font-size:0.83rem;line-height:2;margin:0;padding-left:1.2rem;">
<li><b style="color:#1E293B;">Upload</b> a brain MRI scan (JPG or PNG)</li>
<li>Click <b style="color:#1E293B;">Run AI Analysis</b></li>
<li>Review the predicted <b style="color:#1E293B;">tumor stage</b> and confidence score</li>
<li>Examine the <b style="color:#1E293B;">Grad-CAM heatmap</b> — which regions drove the decision</li>
<li>Cases below confidence threshold are <b style="color:#DC2626;">flagged for review</b></li>
<li>Download the full <b style="color:#1E293B;">clinical report</b></li>
</ol></div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<p style="font-weight:700;color:#1E293B;margin-bottom:0.2rem;">📦 Batch Image Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#64748B;margin-bottom:0.8rem;">Analyze up to 20 images at once. Export results as CSV.</p>', unsafe_allow_html=True)

    batch_files = st.file_uploader("", type=["jpg","jpeg","png"],
                                   accept_multiple_files=True, key="batch",
                                   label_visibility="collapsed")
    if batch_files:
        n = min(len(batch_files), 20)
        batch_files = batch_files[:20]
        ci, cb = st.columns([4,1])
        ci.markdown(f'<div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:8px 14px;font-size:0.83rem;color:#15803D;font-weight:600;">✅ {n} image(s) ready</div>', unsafe_allow_html=True)
        run_b = cb.button("▶ Analyze", type="primary", use_container_width=True)

        if run_b:
            if not api_info:
                st.error("API offline.")
            else:
                prog = st.progress(0, "Preparing…")
                ftuples = []
                for f in batch_files:
                    f.seek(0)
                    ftuples.append(("files",(f.name,f.read(),f.type)))
                prog.progress(15,"Sending to API…")
                try:
                    rsp = requests.post(f"{api_url}/predict/batch", files=ftuples, timeout=300)
                    prog.progress(85,"Processing…")
                    if rsp.ok:
                        results = rsp.json()
                        prog.progress(100); prog.empty()
                        lmb = LABELS_4 if num_cls_g <= 4 else LABELS_5
                        avg_c = np.mean([r["confidence"] for r in results])
                        n_rev = sum(1 for r in results if r["requires_review"])
                        cnts  = {}
                        for r in results: cnts[r["stage_id"]] = cnts.get(r["stage_id"],0)+1

                        k1,k2,k3,k4 = st.columns(4)
                        k1.markdown(kpi_html(str(n),"Analyzed","",    "#1D4ED8","#EFF6FF"), unsafe_allow_html=True)
                        k2.markdown(kpi_html(f"{avg_c*100:.1f}%","Avg Confidence","","#15803D","#F0FDF4"), unsafe_allow_html=True)
                        k3.markdown(kpi_html(str(n_rev),"Need Review","","#DC2626" if n_rev else "#15803D","#FEF2F2" if n_rev else "#F0FDF4"), unsafe_allow_html=True)
                        dom  = max(cnts, key=cnts.get) if cnts else 0
                        di   = lmb.get(dom,("?","#888","⚪","","","",""))
                        k4.markdown(kpi_html(f"{di[2]} {di[0].split()[0]}","Most Frequent","",di[1],"#FFFFFF"), unsafe_allow_html=True)

                        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                        rows = []
                        for r in results:
                            info = lmb.get(r["stage_id"],("Unknown","#888","⚪","","","",""))
                            rows.append({"Filename":r["filename"],"Stage":f"{info[2]} {info[0]}",
                                                    :info[3],"Risk":info[4],
                                                     :f"{r['confidence']*100:.1f}%",
                                                 :"⚠️ YES" if r["requires_review"] else "✅ NO"})
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin:0.8rem 0 0.3rem;">Stage Distribution</p>', unsafe_allow_html=True)
                        cdf = pd.DataFrame({lmb.get(i,("Class "+str(i),))[0]:[cnts.get(i,0)] for i in range(num_cls_g)})
                        st.bar_chart(cdf.T, use_container_width=True, height=220)

                        st.download_button("📥 Export CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv", use_container_width=True)
                    else:
                        st.error(f"API error {rsp.status_code}")
                except Exception as e:
                    prog.empty(); st.error(str(e))

with tab3:
    st.markdown('<p style="font-weight:700;color:#1E293B;margin-bottom:0.3rem;">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:9px;
padding:9px 14px;font-size:0.82rem;color:#166534;margin-bottom:0.8rem;">
✅ Evaluated on <b>1,600 independent test images</b> (400 per class) — no data leakage.
</div>""", unsafe_allow_html=True)

    df_p = pd.DataFrame({
                   :["🟢 Stage 0 — No Tumor","🔵 Stage I — Meningioma","🟡 Stage II — Pituitary","🔴 Stage III/IV — Glioma","Macro Average"],
                   :["96%","85%","95%","97%","93%"],
                   :["100%","94%","99%","78%","93%"],
                   :["98%","89%","97%","86%","93%"],
                   :["400","400","400","400","1,600"],
    })
    st.dataframe(df_p, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin-bottom:0.4rem;">Validation → Test Generalization (no overfitting)</p>', unsafe_allow_html=True)
    t1,t2,t3,t4 = st.columns(4)
    t1.markdown(kpi_html("92.4%→92.8%","Accuracy","Stable ✓","#059669","#F0FDF4"), unsafe_allow_html=True)
    t2.markdown(kpi_html("0.9868→0.9863","ROC-AUC","Stable ✓","#1D4ED8","#EFF6FF"), unsafe_allow_html=True)
    t3.markdown(kpi_html("91.4%→92.8%","Recall","Generalizes ✓","#7C3AED","#F5F3FF"), unsafe_allow_html=True)
    t4.markdown(kpi_html("seuil 0.70","Optimal Threshold","97.8% recall ✓","#D97706","#FFFBEB"), unsafe_allow_html=True)

    st.markdown("---")
    base = os.path.join(os.path.dirname(__file__),"..")
    curves = {
                                         :os.path.join(base,"models","tl_4classes_efficientnetb0_history.png"),
                                          :os.path.join(base,"models","cnn_4classes_history.png"),
                                          :os.path.join(base,"models","cnn_baseline_history.png"),
    }
    avail = {k:v for k,v in curves.items() if os.path.exists(v)}
    if avail:
        st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin-bottom:0.3rem;">📉 Training Curves</p>', unsafe_allow_html=True)
        sel = st.selectbox("", list(avail.keys()), label_visibility="collapsed")
        st.image(avail[sel], use_column_width=True)

    rdir = os.path.join(base,"reports")
    roc  = os.path.join(rdir,"tl_4classes_efficientnetb0_final_roc_curves.png")
    cm   = os.path.join(rdir,"tl_4classes_efficientnetb0_final_confusion_matrix.png")
    if os.path.exists(roc) and os.path.exists(cm):
        st.markdown("---")
        st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin-bottom:0.4rem;">Evaluation Plots — Test Set</p>', unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        p1.image(roc, caption="ROC Curves (4 classes)", use_column_width=True)
        p2.image(cm,  caption="Confusion Matrix (1,600 images)", use_column_width=True)

    gdir = os.path.join(rdir,"gradcam")
    if os.path.isdir(gdir):
        gfiles = [f for f in os.listdir(gdir) if f.endswith(".png")][:6]
        if gfiles:
            st.markdown("---")
            st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin-bottom:0.4rem;">🔍 Grad-CAM Gallery</p>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,f in enumerate(gfiles):
                cols[i%3].image(os.path.join(gdir,f),
                                caption=f.replace("_"," ").replace(".png",""),
                                use_column_width=True)

with tab4:
    st.markdown('<p style="font-weight:700;color:#1E293B;margin-bottom:0.6rem;">📄 Evaluation Reports</p>', unsafe_allow_html=True)
    rdir = os.path.join(os.path.dirname(__file__),"..","reports")
    txts = sorted([f for f in os.listdir(rdir) if f.endswith("_metrics.txt")]) if os.path.isdir(rdir) else []

    if txts:
        sel = st.selectbox("", txts, label_visibility="collapsed")
        content = open(os.path.join(rdir,sel),encoding="utf-8").read()
        st.markdown(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:1.2rem 1.5rem;">
<pre style="font-family:'JetBrains Mono','Courier New',monospace;font-size:0.79rem;color:#334155;
white-space:pre-wrap;margin:0;line-height:1.7;">{content}</pre></div>""", unsafe_allow_html=True)
        st.download_button("📥 Download", data=content.encode("utf-8"),
                           file_name=sel, mime="text/plain")
    else:
        st.markdown("""<div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:10px;
padding:11px 15px;font-size:0.82rem;color:#92400E;">
⚠️ No reports yet — run:<br>
<code>python src/evaluate.py --model models/tl_4classes_efficientnetb0_final.keras</code>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.83rem;font-weight:700;color:#1E293B;margin-bottom:0.5rem;">📁 Project Artifacts</p>', unsafe_allow_html=True)
    base = os.path.join(os.path.dirname(__file__),"..")
    items = [
        ("models/tl_4classes_efficientnetb0_final.keras","🧠","Final Model — EfficientNetB0 TL"),
        ("models/autoencoder_best.keras",                "🔧","Autoencoder — Anomaly Detection"),
        ("models/cnn_4classes_best.keras",               "📦","CNN Baseline 4-classes"),
        ("reports/tl_4classes_efficientnetb0_final_metrics.txt","📊","Test Metrics Report"),
        ("reports/tl_4classes_efficientnetb0_final_roc_curves.png","📈","ROC Curves"),
        ("reports/tl_4classes_efficientnetb0_final_confusion_matrix.png","🧮","Confusion Matrix"),
        ("models/tl_4classes_efficientnetb0_threshold.txt","⚙️","Optimal Threshold (0.70)"),
    ]
    rws = ""
    for path,ico,lbl in items:
        full = os.path.join(base,path)
        ok   = os.path.exists(full)
        sz   = f"{os.path.getsize(full)/1024/1024:.1f} MB" if ok and os.path.getsize(full)>100000 else ""
        sc   = "#15803D" if ok else "#DC2626"
        rws += f'<tr><td style="padding:8px 12px;">{ico} <span style="color:#1E293B;">{lbl}</span></td><td style="padding:8px 12px;font-weight:600;color:{sc};">{"✓ Present" if ok else "✗ Missing"}</td><td style="padding:8px 12px;color:#94A3B8;">{sz}</td></tr>'

    st.markdown(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;overflow:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
<thead><tr style="background:#F8FAFC;border-bottom:1px solid #E2E8F0;">
<th style="padding:9px 12px;text-align:left;color:#64748B;font-weight:600;">Artifact</th>
<th style="padding:9px 12px;text-align:left;color:#64748B;font-weight:600;">Status</th>
<th style="padding:9px 12px;text-align:left;color:#64748B;font-weight:600;">Size</th>
</tr></thead><tbody>{rws}</tbody></div>""", unsafe_allow_html=True)

with tab5:
    a1, a2 = st.columns([3,2], gap="large")
    with a1:
        st.markdown("""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1rem;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.5rem;">Project Overview</div>
<p style="font-size:0.86rem;color:#475569;line-height:1.75;margin:0;">
<b style="color:#1E293B;">NeuroAI</b> is a deep learning system for automated brain tumor staging from MRI scans.
It uses a fine-tuned <b style="color:#1D4ED8;">EfficientNetB0</b> backbone with two-phase transfer learning,
Focal Loss, and Grad-CAM XAI explainability. A confidence threshold of 0.70 flags uncertain cases for manual review.
</p></div>

<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1rem;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.6rem;">4-Class WHO Clinical Staging</div>
<table style="width:100%;border-collapse:collapse;font-size:0.81rem;">
<thead><tr style="background:#F8FAFC;">
<th style="padding:7px 10px;text-align:left;color:#64748B;border-bottom:1px solid #E2E8F0;">Stage</th>
<th style="padding:7px 10px;text-align:left;color:#64748B;border-bottom:1px solid #E2E8F0;">Tumor</th>
<th style="padding:7px 10px;text-align:left;color:#64748B;border-bottom:1px solid #E2E8F0;">WHO Grade</th>
<th style="padding:7px 10px;text-align:left;color:#64748B;border-bottom:1px solid #E2E8F0;">Management</th>
</tr></thead><tbody>
<tr><td style="padding:7px 10px;">🟢 Stage 0</td><td style="padding:7px 10px;color:#1E293B;">No Tumor</td><td style="padding:7px 10px;color:#059669;">Control</td><td style="padding:7px 10px;color:#475569;">Routine check-up</td></tr>
<tr style="background:#FAFAFA;"><td style="padding:7px 10px;">🔵 Stage I</td><td style="padding:7px 10px;color:#1E293B;">Meningioma</td><td style="padding:7px 10px;color:#0284C7;">Grade I</td><td style="padding:7px 10px;color:#475569;">Active surveillance</td></tr>
<tr><td style="padding:7px 10px;">🟡 Stage II</td><td style="padding:7px 10px;color:#1E293B;">Pituitary</td><td style="padding:7px 10px;color:#D97706;">Local</td><td style="padding:7px 10px;color:#475569;">Oncology consult</td></tr>
<tr style="background:#FAFAFA;"><td style="padding:7px 10px;">🔴 Stage III/IV</td><td style="padding:7px 10px;color:#1E293B;">Glioma</td><td style="padding:7px 10px;color:#DC2626;font-weight:600;">Grade III–IV</td><td style="padding:7px 10px;color:#DC2626;font-weight:600;">URGENT treatment</td></tr>
</tbody></table>
<div style="font-size:0.71rem;color:#94A3B8;margin-top:0.5rem;">⚕️ III+IV merged: Kaggle lacks WHO molecular grade labels (IDH1/MGMT). IRM alone cannot distinguish them.</div>
</div>

<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem 1.8rem;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.6rem;">Architecture Pipeline</div>
<pre style="font-family:'JetBrains Mono',monospace;font-size:0.73rem;color:#1D4ED8;background:#EFF6FF;
padding:1rem;border-radius:8px;margin:0;line-height:1.85;">MRI Scan (any size)
  ↓  Resize 224×224px  [EfficientNetB0 built-in preprocessing]
EfficientNetB0 (ImageNet · 7.8M params)
  ↓  Phase 1: Feature Extraction   40 ep · lr=1e-3 · frozen
  ↓  Phase 2: Fine-Tuning          25 ep · lr=5e-6 · last 40 layers
GlobalAvgPool → BN → Dense(512) → Dropout(0.4) → Dense(4, softmax)
  ↓  Focal Loss (γ=2) + class weights
  ↓  Threshold sweep → optimal 0.70 → recall 97.8%
  ↓  Grad-CAM XAI  +  AutoEncoder anomaly (MSE)  +  FastAPI</pre>
</div>""", unsafe_allow_html=True)

    with a2:
        st.markdown('<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem;">' +
                                                                                                                                                                   +
            kpi_html("97.8%","Recall @ threshold 0.70","✓ Clinical target achieved","#15803D","#F0FDF4") +
                                             +
            kpi_html("92.8%","Test Accuracy","1,600 images · 4 classes","#1D4ED8","#EFF6FF") +
                                             +
            kpi_html("0.9863","ROC-AUC","Excellent discriminating power","#7C3AED","#F5F3FF") +
                                             +
            kpi_html("0.9268","F2-Score","Weighted recall clinical KPI","#D97706","#FFFBEB") +
                    , unsafe_allow_html=True)

        st.markdown("""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem;margin-top:1rem;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.6rem;">Tech Stack</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;">""" +
  .join(f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:7px;padding:5px 9px;font-size:0.77rem;color:#475569;">{t}</div>'
for t in ["🐍 Python 3.10","🧠 TF 2.15.1","🏗️ EfficientNetB0","⚡ FastAPI","🎨 Streamlit","📊 scikit-learn","🐳 Docker","🔬 Grad-CAM","🔍 AutoEncoder","📊 Focal Loss"]) +
              , unsafe_allow_html=True)

        st.markdown("""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.4rem;margin-top:1rem;">
<div style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.5rem;">Quick Start</div>
<pre style="font-family:'JetBrains Mono',monospace;font-size:0.71rem;color:#1D4ED8;
background:#EFF6FF;padding:10px;border-radius:7px;margin:0;line-height:1.9;"># API
uvicorn api.main:app --port 8000

# Dashboard
streamlit run dashboard/app.py

# Evaluate
python src/evaluate.py

# Docker
cd docker && docker-compose up</pre></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:1.5rem;padding:0.9rem;
background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;font-size:0.76rem;color:#B91C1C;">
⚠️ <b>CLINICAL DISCLAIMER</b> — NeuroAI is for <b>research purposes only</b>.
It does <b>NOT</b> replace professional medical diagnosis.
All results must be validated by a qualified physician. · © 2026 Mohamed Abidi · NeuroAI v4.0
</div>""", unsafe_allow_html=True)
