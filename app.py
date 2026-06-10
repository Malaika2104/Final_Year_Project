import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="HN Cancer Dashboard", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.big-num   { font-size:2.8rem; font-weight:700; line-height:1.1; }
.label-txt { font-size:0.78rem; text-transform:uppercase;
             letter-spacing:.06em; color:#94a3b8; margin-bottom:4px; }
.card      { background:#1e2130; border-radius:14px; padding:22px 26px;
             border-left:5px solid; margin-bottom:6px; }
.card-teal { border-color:#10b981; }
.card-red  { border-color:#ef4444; }
.mod-badge { display:inline-block; padding:3px 10px; border-radius:20px;
             font-size:0.75rem; font-weight:600; margin:3px; }
</style>
""", unsafe_allow_html=True)

# ─── FEATURE LISTS (raw columns from patients.csv) ───────────────────────────
CLINICAL_RAW = [
    'year_of_initial_diagnosis','age_at_initial_diagnosis','sex','smoking_status',
    'primarily_metastasis','days_to_last_information','first_treatment_intent',
    'first_treatment_modality','days_to_first_treatment','adjuvant_treatment_intent',
    'adjuvant_radiotherapy','adjuvant_radiotherapy_modality','adjuvant_systemic_therapy',
    'adjuvant_systemic_therapy_modality','adjuvant_radiochemotherapy'
]
PATHOLOGY_RAW = [
    'primary_tumor_site','pT_stage','pN_stage','grading','hpv_association_p16',
    'number_of_positive_lymph_nodes','number_of_resected_lymph_nodes','perinodal_invasion',
    'lymphovascular_invasion_L','vascular_invasion_V','perineural_invasion_Pn',
    'resection_status','resection_status_carcinoma_in_situ','carcinoma_in_situ',
    'closest_resection_margin_in_cm','histologic_type','infiltration_depth_in_mm'
]

# ─── LOAD MODELS & ASSETS (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading 4-modality Gradient Boosting model…")
def load_model():
    gb_s  = joblib.load("models/gb_survival.pkl")
    gb_r  = joblib.load("models/gb_recurrence.pkl")
    sc    = joblib.load("models/scaler_c.pkl")
    sp    = joblib.load("models/scaler_p.pkl")
    sb    = joblib.load("models/scaler_b.pkl")
    sw    = joblib.load("models/scaler_w.pkl")
    clin_cols  = json.load(open("models/clinical_cols.json"))
    path_cols  = json.load(open("models/pathology_cols.json"))
    blood_cols = json.load(open("models/blood_cols.json"))
    wsi_cols   = json.load(open("models/wsi_cols.json"))
    wsi_df     = pd.read_csv("models/wsi_features.csv")
    return gb_s, gb_r, sc, sp, sb, sw, clin_cols, path_cols, blood_cols, wsi_cols, wsi_df

@st.cache_data
def load_data():
    df = pd.read_csv("patients.csv")
    df.columns = df.columns.str.strip()
    return df

gb_s, gb_r, sc, sp, sb, sw, clin_cols, path_cols, blood_cols, wsi_cols, wsi_df = load_model()
df = load_data()
blood_cols_clean = [c for c in blood_cols if c in df.columns]

# ─── ALL FEATURE NAMES (for importance mapping) ───────────────────────────────
ALL_FEAT_NAMES = clin_cols + path_cols + blood_cols + wsi_cols

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict(p):
    # Clinical
    clin_raw = pd.DataFrame([p[CLINICAL_RAW]])
    clin_d   = pd.get_dummies(clin_raw).reindex(columns=clin_cols, fill_value=0).astype(float).fillna(0)

    # Pathology
    path_raw = pd.DataFrame([p[PATHOLOGY_RAW]])
    path_d   = pd.get_dummies(path_raw).reindex(columns=path_cols, fill_value=0).astype(float).fillna(0)

    # Blood
    blood_d  = pd.DataFrame([p[blood_cols_clean]]).reindex(columns=blood_cols, fill_value=0).astype(float).fillna(0)

    # WSI
    wsi_row  = wsi_df[wsi_df['patient_id'] == p['patient_id']]
    if len(wsi_row) > 0:
        wsi_d = wsi_row[wsi_cols].astype(float).fillna(0).values
        has_wsi = True
    else:
        wsi_d   = np.zeros((1, len(wsi_cols)))
        has_wsi = False

    # Scale & combine
    Xc    = sc.transform(clin_d)
    Xp    = sp.transform(path_d)
    Xb    = sb.transform(blood_d)
    Xw    = sw.transform(wsi_d)
    X_all = np.hstack([Xc, Xp, Xb, Xw])

    surv_prob = gb_s.predict_proba(X_all)[0]
    rec_prob  = gb_r.predict_proba(X_all)[0]

    living_pct = round((1 - surv_prob[1]) * 100, 1)
    rec_pct    = round(rec_prob[1] * 100, 1)

    # Feature importance — top 10 per task, label by modality
    def get_top(fi, n=10):
        idx = np.argsort(fi)[::-1][:n]
        result = []
        for i in idx:
            name = ALL_FEAT_NAMES[i] if i < len(ALL_FEAT_NAMES) else f"feature_{i}"
            # Tag modality
            if i < len(clin_cols):                              tag = "🔵 Clinical"
            elif i < len(clin_cols)+len(path_cols):            tag = "🟣 Pathology"
            elif i < len(clin_cols)+len(path_cols)+len(blood_cols): tag = "🔴 Blood"
            else:                                               tag = "🟡 WSI"
            result.append((name, round(fi[i]*100, 2), tag))
        return result

    top_s = get_top(gb_s.feature_importances_)
    top_r = get_top(gb_r.feature_importances_)

    return living_pct, rec_pct, top_s, top_r, has_wsi

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("## 🧬 Head & Neck Cancer Prognostic Dashboard")
st.caption("FYP2025-HealthTech · UET Peshawar · HANCOCK Dataset · **4-Modality Gradient Boosting**")

# Modality badges
st.markdown("""
<span class="mod-badge" style="background:#1e3a5f;color:#60a5fa">🔵 Clinical</span>
<span class="mod-badge" style="background:#3b1f5e;color:#c084fc">🟣 Pathological</span>
<span class="mod-badge" style="background:#5e1f1f;color:#f87171">🔴 Blood</span>
<span class="mod-badge" style="background:#3d3a1a;color:#fbbf24">🟡 WSI (Histology)</span>
""", unsafe_allow_html=True)
st.divider()

# ─── TOP METRICS ─────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Patients", 763)
m2.metric("Avg Age", round(df["age_at_initial_diagnosis"].mean(), 1))
m3.metric("Survival Rate", f"{round(df['survival_status'].eq('living').mean()*100,1)}%")
m4.metric("Recurrence Rate", f"{round(df['recurrence'].eq('yes').mean()*100,1)}%")
m5.metric("Model Features", "1,181")
st.divider()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Patient Lookup")
st.sidebar.caption("Enter Patient ID for 4-modality prediction")
pid = st.sidebar.text_input("Patient ID", placeholder="e.g. 1, 42, 250")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Model Info**")
st.sidebar.markdown("Gradient Boosting Classifier")
st.sidebar.markdown("4 Modalities · 1,181 Features")
st.sidebar.markdown(f"Training samples: 610")
st.sidebar.markdown(f"Survival Acc: **71.9%**")
st.sidebar.markdown(f"Recurrence Acc: **81.0%**")
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Dataset**")
st.sidebar.write(f"Living: {df['survival_status'].eq('living').sum()}")
st.sidebar.write(f"Deceased: {df['survival_status'].eq('deceased').sum()}")
st.sidebar.write(f"Recurrence Yes: {df['recurrence'].eq('yes').sum()}")
st.sidebar.write(f"Recurrence No: {df['recurrence'].eq('no').sum()}")

# ─── PATIENT RESULTS ─────────────────────────────────────────────────────────
if pid:
    pid = pid.strip()
    patient = df[df["patient_id"].astype(str) == pid]

    if patient.empty:
        st.error(f"❌ Patient ID **{pid}** not found. Valid IDs: 1 – {len(df)}")
    else:
        p = patient.iloc[0]
        with st.spinner("Running 4-modality prediction…"):
            living_pct, rec_pct, top_s, top_r, has_wsi = predict(p)

        st.subheader(f"👤 Patient {pid} — Prediction Results")

        # WSI warning if missing
        if not has_wsi:
            st.warning("⚠️ No WSI data found for this patient — prediction uses Clinical + Pathological + Blood only.")

        # ── Prediction cards ───────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card card-teal">
                <div class="label-txt">Survival Probability</div>
                <div class="big-num" style="color:#10b981">{living_pct}%</div>
                <div style="color:#64748b;font-size:.88rem;margin-top:6px">
                    Probability patient is <b>living</b> at last follow-up
                </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            rc = "#10b981" if rec_pct < 20 else "#f59e0b" if rec_pct <= 45 else "#ef4444"
            st.markdown(f"""
            <div class="card card-red">
                <div class="label-txt">Recurrence Risk</div>
                <div class="big-num" style="color:{rc}">{rec_pct}%</div>
                <div style="color:#64748b;font-size:.88rem;margin-top:6px">
                    Probability of cancer <b>recurrence</b> after treatment
                </div>
            </div>""", unsafe_allow_html=True)

        # Risk badge
        if rec_pct < 20:
            st.success("🟢 LOW RECURRENCE RISK")
        elif rec_pct <= 45:
            st.warning("🟡 MODERATE RECURRENCE RISK")
        else:
            st.error("🔴 HIGH RECURRENCE RISK")

        st.divider()

        # ── Feature importance ─────────────────────────────────────────────
        fi1, fi2 = st.columns(2)

        def render_bars(features, accent):
            max_v = features[0][1] if features else 1
            for name, score, tag in features:
                w = int((score / max_v) * 100)
                st.markdown(f"""
                <div style="margin-bottom:9px">
                  <div style="display:flex;justify-content:space-between;
                              font-size:.82rem;color:#cbd5e0;margin-bottom:3px">
                    <span>{tag} &nbsp;<b>{name}</b></span>
                    <span style="color:{accent}">{score:.1f}%</span>
                  </div>
                  <div style="background:#2d3748;border-radius:4px;height:10px">
                    <div style="width:{w}%;background:{accent};
                                height:10px;border-radius:4px;opacity:0.85"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with fi1:
            st.markdown("##### 📊 Top Factors → Survival Prediction")
            render_bars(top_s, "#10b981")

        with fi2:
            st.markdown("##### 📊 Top Factors → Recurrence Prediction")
            render_bars(top_r, "#ef4444")

        st.divider()

        # ── Clinical Summary ───────────────────────────────────────────────
        st.subheader("🧾 Clinical Summary")
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("**🔵 Patient Info**")
            st.write(f"🎂 Age: **{p.get('age_at_initial_diagnosis','N/A')}**")
            st.write(f"⚧ Sex: **{p.get('sex','N/A')}**")
            st.write(f"🚬 Smoking: **{p.get('smoking_status','N/A')}**")
            st.write(f"📅 Year: **{p.get('year_of_initial_diagnosis','N/A')}**")

        with s2:
            st.markdown("**🟣 Tumor Info**")
            st.write(f"📍 Site: **{p.get('primary_tumor_site','N/A')}**")
            st.write(f"🔬 pT Stage: **{p.get('pT_stage','N/A')}**")
            st.write(f"🔗 pN Stage: **{p.get('pN_stage','N/A')}**")
            st.write(f"⭐ Grading: **{p.get('grading','N/A')}**")
            st.write(f"🧫 HPV/p16: **{p.get('hpv_association_p16','N/A')}**")

        with s3:
            st.markdown("**✅ Actual Outcomes**")
            surv_a = p.get('survival_status','N/A')
            rec_a  = p.get('recurrence','N/A')
            st.write(f"{'✅' if surv_a=='living' else '⚠️'} Status: **{surv_a}**")
            st.write(f"{'⚠️' if rec_a=='yes' else '✅'} Recurrence: **{rec_a}**")
            st.write(f"🔪 Resection: **{p.get('resection_status','N/A')}**")
            st.write(f"🔴 Metastasis: **{p.get('primarily_metastasis','N/A')}**")

        st.divider()

        # ── Blood + WSI expanders ──────────────────────────────────────────
        with st.expander("🔴 Blood Test Values", expanded=False):
            bdata = {c: p.get(c, np.nan) for c in blood_cols_clean}
            bdf   = pd.DataFrame(list(bdata.items()), columns=["Marker","Value"])
            bdf["Value"] = bdf["Value"].apply(lambda x: round(float(x),3) if pd.notna(x) else "N/A")
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        with st.expander("🟡 WSI Features (top 10 by importance)", expanded=False):
            wsi_row = wsi_df[wsi_df['patient_id']==p['patient_id']]
            if len(wsi_row) > 0:
                wsi_imp = sorted(zip(wsi_cols, gb_s.feature_importances_[len(clin_cols)+len(path_cols)+len(blood_cols):]),
                                 key=lambda x: -x[1])[:10]
                wsi_show = pd.DataFrame(wsi_imp, columns=["WSI Feature","Importance"])
                wsi_show["Value"] = [round(float(wsi_row[f].values[0]),4) for f,_ in wsi_imp]
                st.dataframe(wsi_show, use_container_width=True, hide_index=True)
            else:
                st.info("No WSI data for this patient.")

        with st.expander("📄 Full Patient Record", expanded=False):
            st.dataframe(patient, use_container_width=True)

else:
    st.info("👈 Enter a **Patient ID** in the sidebar (e.g. 1, 42, 150) to see 4-modality predictions.")
    tab1, tab2 = st.tabs(["📋 Dataset Preview", "📊 Distribution"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            sv = df['survival_status'].value_counts().reset_index()
            sv.columns = ['Status','Count']
            st.markdown("**Survival Status**")
            st.dataframe(sv, use_container_width=True, hide_index=True)
        with col_b:
            rv = df['recurrence'].value_counts().reset_index()
            rv.columns = ['Recurrence','Count']
            st.markdown("**Recurrence**")
            st.dataframe(rv, use_container_width=True, hide_index=True)

st.divider()
st.caption("🧬 FYP2025-HealthTech · UET Peshawar · HANCOCK Dataset · 4-Modality Gradient Boosting (Clinical + Pathological + Blood + WSI)")
