import streamlit as st
import pandas as pd
import pickle
import sys

# Page config
st.set_page_config(layout="wide")

# ================= STYLE =================
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = pickle.load(open(r"C:\Users\priya\Downloads\Infotectproject1\model.pkl", "rb"))

# ================= HEADER =================
st.markdown("<h1 style='text-align: center; color: #00BFFF;'>🏨 Booking Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI Powered Hotel Analytics System</p>", unsafe_allow_html=True)

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload your dataset")

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ================= FILTER =================
    hotel_filter = st.selectbox("Select Hotel Type", df['hotel'].unique())
    df = df[df['hotel'] == hotel_filter]

    # ================= KPIs =================
    total_bookings = len(df)
    total_cancellations = df['is_canceled'].sum()
    avg_adr = df['adr'].mean()
    cancel_rate = total_cancellations / total_bookings

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Bookings", total_bookings)
    c2.metric("Total Cancellations", int(total_cancellations))
    c3.metric("Average ADR", round(avg_adr, 2))
    c4.metric("Cancellation Rate", round(cancel_rate, 2))

    # ================= CHARTS =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Type vs Cancellation")
        data1 = df.groupby('customer_type')['is_canceled'].sum().sort_values()
        st.bar_chart(data1)

    with col2:
        st.subheader("Market Segment vs Cancellation")
        data2 = df.groupby('market_segment')['is_canceled'].sum().sort_values()
        st.bar_chart(data2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Hotel Type vs Cancellation")
        data3 = df.groupby('hotel')['is_canceled'].sum()
        st.bar_chart(data3)

    with col4:
        st.subheader("ADR Trend by Month")
        data4 = df.groupby('arrival_date_month')['adr'].mean()
        st.line_chart(data4)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Monthly Bookings")
        st.line_chart(df['arrival_date_month'].value_counts())

    with col6:
        st.subheader("Monthly Cancellations")
        st.line_chart(df[df['is_canceled'] == 1]['arrival_date_month'].value_counts())

    # ================= INSIGHTS =================
    st.subheader("📌 Key Insights")

    st.info("📌 High lead time leads to higher cancellations")
    st.info("📌 City hotels have more cancellations")
    st.info("📌 Online bookings show higher cancellation risk")

# ================= ML PREDICTION =================
st.subheader("🤖 AI Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    lead_time = st.slider("Lead Time", 0, 500, 50)

with col2:
    adr = st.slider("ADR", 0, 500, 100)

with col3:
    total_stay = st.slider("Total Stay", 1, 30, 3)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'lead_time': [lead_time],
        'adr': [adr],
        'total_stay': [total_stay]
    })

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if pred[0] == 1:
        st.error("❌ High chance of cancellation")
        st.warning("💡 Suggestion: Take advance payment")
    else:
        st.success("✅ Booking likely confirmed")
        st.success("💡 Suggestion: Offer loyalty benefits")

    st.info(f"📊 Cancellation Probability: {round(prob*100,2)}%")
