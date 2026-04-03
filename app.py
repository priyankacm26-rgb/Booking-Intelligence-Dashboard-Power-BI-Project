import streamlit as st
import pandas as pd
import pickle
import sys

st.set_page_config(layout="wide")

# ================= PREMIUM UI =================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00BFFF;
    text-align: center;
}
[data-testid="metric-container"] {
    background-color: #1c1f26;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
.stButton>button {
    background-color: #00BFFF;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))
# ================= HEADER =================
st.markdown("<h1>🏨 Booking Intelligence Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; color:gray;'>
👩‍💻 Developed by <b>Priyanka C Meti</b> & <b>Dharanendra</b>
</p>
""", unsafe_allow_html=True)

# ================= FILE LOAD =================
file = st.file_uploader("Upload dataset (optional)")

if file:
    df = pd.read_csv(file)
else:
 pd.read_csv("cleaned_hotel_booking.csv")
# ================= SIDEBAR FILTER =================
st.sidebar.header("🔍 Filters")
hotel = st.sidebar.selectbox("Hotel Type", df['hotel'].unique())
df = df[df['hotel'] == hotel]

# ================= DATA PREVIEW =================
st.subheader("📊 Data Preview")
st.dataframe(df.head())

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

    st.subheader("📊 Prediction Result")

    st.info(f"Cancellation Probability: {round(prob*100,2)}%")

    if prob > 0.7:
        st.error("🔴 High Risk Booking")
        st.warning("💡 Action: Take advance payment + reminders")

    elif prob > 0.4:
        st.warning("🟡 Medium Risk Booking")
        st.info("💡 Action: Offer discounts")

    else:
        st.success("🟢 Low Risk Booking")
        st.success("💡 Safe booking")

    # ================= MODEL INSIGHT =================
    st.subheader("📌 Model Insight")

    importance = pd.DataFrame({
        "Feature": ["Lead Time", "ADR", "Total Stay"],
        "Impact": model.coef_[0]
    })

    st.bar_chart(importance.set_index("Feature"))
