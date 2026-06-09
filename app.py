import streamlit as st
import pandas as pd
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Head & Neck Cancer Dashboard",
    layout="wide"
)

st.title("🧬 Head & Neck Cancer Prognostic Dashboard")

# ======================
# LOAD DATA
# ======================
# change file name if needed
df = pd.read_csv("patients.csv")

# ======================
# SIDEBAR - PATIENT SEARCH
# ======================
st.sidebar.header("🔍 Patient Search")

patient_id = st.sidebar.text_input("Enter Patient ID")

# ======================
# TOP METRICS
# ======================
col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(df))
col2.metric("Avg Age", round(df["Age"].mean(), 1))

if "Recurrence" in df.columns:
    col3.metric("Recurrence Rate", f"{round(df['Recurrence'].mean()*100,1)}%")
else:
    col3.metric("Recurrence Rate", "N/A")

st.divider()

# ======================
# DATA PREVIEW
# ======================
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

st.divider()

# ======================
# PATIENT DETAILS
# ======================
if patient_id:

    patient = df[df["Patient_ID"] == patient_id]

    if not patient.empty:
        st.subheader("👤 Patient Details")
        st.write(patient)

        patient = patient.iloc[0]

        st.divider()

        # ======================
        # PLACEHOLDER PREDICTIONS
        # (Replace later with ML model)
        # ======================

        st.subheader("📈 Predictions")

        # fake logic (replace with model later)
        survival_prob = max(10, 100 - patient["Age"] - 20)
        recurrence_risk = min(90, patient["Age"] + 10)

        col1, col2 = st.columns(2)

        col1.metric("Survival Probability", f"{survival_prob:.2f}%")
        col2.metric("Recurrence Risk", f"{recurrence_risk:.2f}%")

        # ======================
        # RISK CATEGORY
        # ======================
        st.subheader("⚠ Risk Category")

        if recurrence_risk < 30:
            st.success("LOW RISK")
        elif recurrence_risk <= 60:
            st.warning("MEDIUM RISK")
        else:
            st.error("HIGH RISK")

        st.divider()

        # ======================
        # IMAGE DISPLAY
        # ======================
        st.subheader("🖼 Medical Image")

        img_folder = "images"
        img_path = os.path.join(img_folder, patient_id + ".jpg")

        if os.path.exists(img_path):
            st.image(img_path, caption="Patient Scan")
        else:
            st.warning("Image not found for this patient")

    else:
        st.error("Patient ID not found")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Built for Final Year Project – Head & Neck Cancer Prognostic System")