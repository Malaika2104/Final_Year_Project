import streamlit as st
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Cancer Dashboard", layout="wide")

st.title("🧬 Head & Neck Cancer Prognostic Dashboard")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("patients.csv")

# =========================
# CLEAN COLUMN SAFETY
# =========================
df.columns = df.columns.str.strip()

# =========================
# TOP METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(df))

# Safe age calculation
if "age_at_initial_diagnosis" in df.columns:
    col2.metric("Avg Age", round(df["age_at_initial_diagnosis"].mean(), 1))
else:
    col2.metric("Avg Age", "N/A")

col3.metric("Total Features", df.shape[1])

st.divider()

# =========================
# PATIENT SEARCH
# =========================
st.sidebar.header("🔍 Patient Search")

patient_id = st.sidebar.text_input("Enter Patient ID (e.g. 01, 02)")

if patient_id:

    patient = df[df["patient_id"].astype(str) == str(patient_id)]

    if not patient.empty:

        st.subheader("👤 Patient Details")
        st.dataframe(patient)

        p = patient.iloc[0]

        st.divider()

        # =========================
        # BASIC RISK LOGIC (TEMP)
        # =========================
        age = p.get("age_at_initial_diagnosis", 50)

        smoking = p.get("smoking_status", "unknown")

        # simple scoring logic (replace later with ML)
        survival_score = max(10, 100 - age)

        recurrence_score = age + (10 if smoking == "smoker" else 0)

        col1, col2 = st.columns(2)

        col1.metric("Survival Probability", f"{survival_score:.1f}%")
        col2.metric("Recurrence Risk", f"{recurrence_score:.1f}%")

        # =========================
        # RISK CATEGORY
        # =========================
        st.subheader("⚠ Risk Category")

        if recurrence_score < 30:
            st.success("LOW RISK 🟢")
        elif recurrence_score <= 60:
            st.warning("MEDIUM RISK 🟡")
        else:
            st.error("HIGH RISK 🔴")

        st.divider()

        # =========================
        # KEY CLINICAL INFO
        # =========================
        st.subheader("🧾 Clinical Summary")

        st.write("Age:", p.get("age_at_initial_diagnosis", "N/A"))
        st.write("Sex:", p.get("sex", "N/A"))
        st.write("Smoking:", p.get("smoking_status", "N/A"))
        st.write("Tumor Stage (pT):", p.get("pT_stage", "N/A"))
        st.write("Node Stage (pN):", p.get("pN_stage", "N/A"))
        st.write("Recurrence:", p.get("recurrence", "N/A"))

    else:
        st.error("❌ Patient ID not found")

# =========================
# DATA PREVIEW
# =========================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# FOOTER
# =========================
st.caption("Final Year Project | Cancer Prognostic System")