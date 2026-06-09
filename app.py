import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Head & Neck Cancer Dashboard",
    page_icon="🧬",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262b3d);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .survival-card { border-color: #00c9a7; }
    .recurrence-card { border-color: #ff6b6b; }
    .feature-bar {
        height: 22px;
        border-radius: 4px;
        background: linear-gradient(90deg, #00c9a7, #007bff);
        margin-bottom: 6px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD & TRAIN MODEL (cached)
# =========================
CLINICAL_FEATURES = [
    'age_at_initial_diagnosis', 'sex', 'smoking_status', 'primarily_metastasis',
    'pT_stage', 'pN_stage', 'grading', 'hpv_association_p16',
    'number_of_positive_lymph_nodes', 'number_of_resected_lymph_nodes',
    'perinodal_invasion', 'lymphovascular_invasion_L', 'vascular_invasion_V',
    'perineural_invasion_Pn', 'resection_status', 'primary_tumor_site',
    'infiltration_depth_in_mm', 'histologic_type'
]
BLOOD_FEATURES = [
    'Basophils', 'CRP', 'Calcium', 'Chloride', 'Creatinine',
    'Eosinophils', 'Erythrocytes', 'Hemoglobin', 'Leukocytes',
    'Lymphocytes', 'Platelets', 'Sodium', 'Potassium'
]
ALL_FEATURES = CLINICAL_FEATURES + BLOOD_FEATURES

FEATURE_LABELS = {
    'age_at_initial_diagnosis': 'Age at Diagnosis',
    'sex': 'Sex',
    'smoking_status': 'Smoking Status',
    'primarily_metastasis': 'Primary Metastasis',
    'pT_stage': 'Tumor Stage (pT)',
    'pN_stage': 'Node Stage (pN)',
    'grading': 'Tumor Grading',
    'hpv_association_p16': 'HPV/p16 Status',
    'number_of_positive_lymph_nodes': 'Positive Lymph Nodes',
    'number_of_resected_lymph_nodes': 'Resected Lymph Nodes',
    'perinodal_invasion': 'Perinodal Invasion',
    'lymphovascular_invasion_L': 'Lymphovascular Invasion',
    'vascular_invasion_V': 'Vascular Invasion',
    'perineural_invasion_Pn': 'Perineural Invasion',
    'resection_status': 'Resection Status',
    'primary_tumor_site': 'Primary Tumor Site',
    'infiltration_depth_in_mm': 'Infiltration Depth (mm)',
    'histologic_type': 'Histologic Type',
    'Basophils': 'Basophils',
    'CRP': 'CRP',
    'Calcium': 'Calcium',
    'Chloride': 'Chloride',
    'Creatinine': 'Creatinine',
    'Eosinophils': 'Eosinophils',
    'Erythrocytes': 'Erythrocytes',
    'Hemoglobin': 'Hemoglobin',
    'Leukocytes': 'Leukocytes',
    'Lymphocytes': 'Lymphocytes',
    'Platelets': 'Platelets',
    'Sodium': 'Sodium',
    'Potassium': 'Potassium',
}

@st.cache_resource
def load_and_train():
    df = pd.read_csv("patients.csv")
    df.columns = df.columns.str.strip()

    X = df[ALL_FEATURES].copy()

    # Encode categoricals
    le_dict = {}
    cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Impute
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    y_surv = (df['survival_status'] == 'deceased').astype(int)
    y_rec  = (df['recurrence'] == 'yes').astype(int)

    rf_surv = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_surv.fit(X_imp, y_surv)

    rf_rec = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_rec.fit(X_imp, y_rec)

    return df, rf_surv, rf_rec, le_dict, imp

df, rf_surv, rf_rec, le_dict, imp = load_and_train()

# =========================
# PREDICT FUNCTION
# =========================
def predict_patient(patient_row):
    xp = patient_row[ALL_FEATURES].copy()
    xp_df = pd.DataFrame([xp])

    cat_cols = xp_df.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        xp_df[col] = xp_df[col].astype(str)
        if col in le_dict:
            try:
                xp_df[col] = le_dict[col].transform(xp_df[col])
            except ValueError:
                xp_df[col] = 0
        else:
            xp_df[col] = 0

    xp_imp = imp.transform(xp_df)

    surv_prob  = rf_surv.predict_proba(xp_imp)[0]   # [deceased, living] → index 0 = deceased prob
    rec_prob   = rf_rec.predict_proba(xp_imp)[0]    # [no, yes] → index 1 = yes prob

    survival_pct   = round(surv_prob[0] * 100, 1)  # probability of being deceased
    living_pct     = round(100 - survival_pct, 1)
    recurrence_pct = round(rec_prob[1] * 100, 1)

    # Per-feature importance × patient feature value (normalized contribution)
    surv_fi  = rf_surv.feature_importances_
    rec_fi   = rf_rec.feature_importances_

    feat_contrib_surv = [(FEATURE_LABELS.get(f, f), round(i * 100, 2))
                         for f, i in sorted(zip(ALL_FEATURES, surv_fi), key=lambda x: -x[1])[:10]]
    feat_contrib_rec  = [(FEATURE_LABELS.get(f, f), round(i * 100, 2))
                         for f, i in sorted(zip(ALL_FEATURES, rec_fi), key=lambda x: -x[1])[:10]]

    return living_pct, recurrence_pct, feat_contrib_surv, feat_contrib_rec

# =========================
# HEADER
# =========================
st.markdown("## 🧬 Head & Neck Cancer Prognostic Dashboard")
st.caption("Final Year Project · FYP2025-HealthTech · UET Peshawar · HANCOCK Dataset")
st.divider()

# =========================
# TOP METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Patients", len(df))
m2.metric("Avg Age", round(df["age_at_initial_diagnosis"].mean(), 1))
m3.metric("Survival Rate",
          f"{round(df['survival_status'].eq('living').mean()*100, 1)}%")
m4.metric("Recurrence Rate",
          f"{round(df['recurrence'].eq('yes').mean()*100, 1)}%")

st.divider()

# =========================
# SIDEBAR — PATIENT SEARCH
# =========================
st.sidebar.header("🔍 Patient Lookup")
st.sidebar.caption("Enter a patient ID to see their predicted outcomes")

patient_id_input = st.sidebar.text_input("Patient ID", placeholder="e.g. 1, 42, 100")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Quick Stats")
st.sidebar.write(f"**Total Features:** {df.shape[1]}")
st.sidebar.write(f"**Living patients:** {df['survival_status'].eq('living').sum()}")
st.sidebar.write(f"**Deceased patients:** {df['survival_status'].eq('deceased').sum()}")
st.sidebar.write(f"**Recurrence (yes):** {df['recurrence'].eq('yes').sum()}")
st.sidebar.write(f"**Recurrence (no):** {df['recurrence'].eq('no').sum()}")

# =========================
# PATIENT RESULTS
# =========================
if patient_id_input:
    patient_id_input = patient_id_input.strip()
    patient = df[df["patient_id"].astype(str) == patient_id_input]

    if patient.empty:
        st.error(f"❌ Patient ID **{patient_id_input}** not found. Try IDs from 1 to {len(df)}.")
    else:
        p = patient.iloc[0]

        # Run prediction
        with st.spinner("Running model predictions..."):
            living_pct, rec_pct, fi_surv, fi_rec = predict_patient(p)

        st.subheader(f"👤 Patient {patient_id_input} — Prediction Results")

        # ── PREDICTION CARDS ──────────────────────────────
        col_s, col_r = st.columns(2)

        with col_s:
            st.markdown(f"""
            <div class="metric-card survival-card">
                <div style="color:#a0aec0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em">
                    Survival Probability
                </div>
                <div style="font-size:3rem; font-weight:700; color:#00c9a7; margin: 6px 0">
                    {living_pct}%
                </div>
                <div style="color:#718096; font-size:0.9rem">
                    Probability patient is <b>living</b> at last follow-up
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            rec_color = "#ff6b6b" if rec_pct > 40 else "#f6ad55" if rec_pct > 20 else "#68d391"
            st.markdown(f"""
            <div class="metric-card recurrence-card">
                <div style="color:#a0aec0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em">
                    Recurrence Risk
                </div>
                <div style="font-size:3rem; font-weight:700; color:{rec_color}; margin: 6px 0">
                    {rec_pct}%
                </div>
                <div style="color:#718096; font-size:0.9rem">
                    Probability of cancer <b>recurrence</b> after treatment
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RISK BADGE ────────────────────────────────────
        if rec_pct < 20:
            st.success("🟢 LOW RECURRENCE RISK")
        elif rec_pct <= 40:
            st.warning("🟡 MODERATE RECURRENCE RISK")
        else:
            st.error("🔴 HIGH RECURRENCE RISK")

        st.divider()

        # ── FEATURE IMPORTANCE CHARTS ─────────────────────
        fi_col1, fi_col2 = st.columns(2)

        with fi_col1:
            st.markdown('<div class="section-header">📊 Top Factors → Survival Prediction</div>',
                        unsafe_allow_html=True)
            max_s = fi_surv[0][1] if fi_surv else 1
            for feat, score in fi_surv:
                bar_width = int((score / max_s) * 100)
                st.markdown(f"""
                <div style="margin-bottom:8px">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#cbd5e0; margin-bottom:3px">
                        <span>{feat}</span><span style="color:#00c9a7">{score:.1f}%</span>
                    </div>
                    <div style="background:#2d3748; border-radius:4px; height:10px">
                        <div style="width:{bar_width}%; background:linear-gradient(90deg,#00c9a7,#007bff);
                                    height:10px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with fi_col2:
            st.markdown('<div class="section-header">📊 Top Factors → Recurrence Prediction</div>',
                        unsafe_allow_html=True)
            max_r = fi_rec[0][1] if fi_rec else 1
            for feat, score in fi_rec:
                bar_width = int((score / max_r) * 100)
                st.markdown(f"""
                <div style="margin-bottom:8px">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#cbd5e0; margin-bottom:3px">
                        <span>{feat}</span><span style="color:#ff6b6b">{score:.1f}%</span>
                    </div>
                    <div style="background:#2d3748; border-radius:4px; height:10px">
                        <div style="width:{bar_width}%; background:linear-gradient(90deg,#ff6b6b,#ed8936);
                                    height:10px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── CLINICAL SUMMARY ──────────────────────────────
        st.subheader("🧾 Clinical Summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Patient Info**")
            st.write(f"🎂 Age: **{p.get('age_at_initial_diagnosis', 'N/A')}**")
            st.write(f"⚧ Sex: **{p.get('sex', 'N/A')}**")
            st.write(f"🚬 Smoking: **{p.get('smoking_status', 'N/A')}**")
            st.write(f"📅 Diagnosed: **{p.get('year_of_initial_diagnosis', 'N/A')}**")

        with c2:
            st.markdown("**Tumor Info**")
            st.write(f"📍 Site: **{p.get('primary_tumor_site', 'N/A')}**")
            st.write(f"🔬 pT Stage: **{p.get('pT_stage', 'N/A')}**")
            st.write(f"🔗 pN Stage: **{p.get('pN_stage', 'N/A')}**")
            st.write(f"⭐ Grading: **{p.get('grading', 'N/A')}**")
            st.write(f"🧫 HPV/p16: **{p.get('hpv_association_p16', 'N/A')}**")

        with c3:
            st.markdown("**Outcomes**")
            surv_actual = p.get('survival_status', 'N/A')
            rec_actual  = p.get('recurrence', 'N/A')
            surv_emoji  = "✅" if surv_actual == "living" else "⚠️"
            rec_emoji   = "⚠️" if rec_actual == "yes" else "✅"
            st.write(f"{surv_emoji} Actual Status: **{surv_actual}**")
            st.write(f"{rec_emoji} Actual Recurrence: **{rec_actual}**")
            st.write(f"🔪 Resection: **{p.get('resection_status', 'N/A')}**")
            st.write(f"🔴 Metastasis: **{p.get('primarily_metastasis', 'N/A')}**")

        st.divider()

        # ── BLOOD VALUES ──────────────────────────────────
        with st.expander("🩸 Blood Test Values", expanded=False):
            blood_data = {FEATURE_LABELS.get(f, f): p.get(f, np.nan) for f in BLOOD_FEATURES}
            blood_df = pd.DataFrame(list(blood_data.items()), columns=["Marker", "Value"])
            blood_df["Value"] = blood_df["Value"].apply(
                lambda x: round(float(x), 3) if pd.notna(x) else "N/A"
            )
            st.dataframe(blood_df, use_container_width=True, hide_index=True)

        # ── RAW ROW ───────────────────────────────────────
        with st.expander("📄 Full Patient Record", expanded=False):
            st.dataframe(patient, use_container_width=True)

else:
    # Show instructions when no patient selected
    st.info("👈 Enter a **Patient ID** in the sidebar (e.g. 1, 42, 150) to see survival & recurrence predictions.")

    # ── DATASET OVERVIEW ─────────────────────────────────
    st.subheader("📊 Dataset Overview")

    tab1, tab2 = st.tabs(["Preview", "Distribution"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            surv_counts = df['survival_status'].value_counts().reset_index()
            surv_counts.columns = ['Status', 'Count']
            st.markdown("**Survival Status**")
            st.dataframe(surv_counts, use_container_width=True, hide_index=True)
        with col_d2:
            rec_counts = df['recurrence'].value_counts().reset_index()
            rec_counts.columns = ['Recurrence', 'Count']
            st.markdown("**Recurrence**")
            st.dataframe(rec_counts, use_container_width=True, hide_index=True)

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🧬 FYP2025-HealthTech · Head & Neck Cancer Outcome Prediction · UET Peshawar · HANCOCK Dataset")
