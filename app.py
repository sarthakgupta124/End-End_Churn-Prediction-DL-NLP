import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictionPipeline

st.set_page_config(page_title="Churn Scorer Pro", page_icon="📉", layout="wide")

st.title("🛡️ Customer Churn Analyzer")
st.caption("Deep Learning Modular Inference Engine")

@st.cache_resource
def get_pipeline():
    return PredictionPipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"System Offline: {e}")
    st.stop()

# Expanded Feedback Options (Top 10 categories + specific sentiments)
review_options = [
    "No reason specified", "Poor Product Quality", "Too many ads", 
    "Poor Customer Service", "Poor Website", "Quality Customer Care",
    "User Friendly Website", "Products always in Stock", "Reasonable Price",
    "Great discounts", "Slow delivery", "Excellent support", "High subscription cost",
    "App crashes frequently", "Easy returns", "Interface is confusing",
    "Slow response time", "High shipping fees", "Love the loyalty program",
    "Limited payment options", "Mobile app is great", "Billing issues",
    "Better rewards needed", "Responsive chat support", "Product was damaged"
    # Note: These map to your training vectors via Word2Vec logic
]

with st.form("churn_input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profile")
        age = st.number_input("Age", 18, 95, 30)
        gender = st.selectbox("Gender", ["M", "F"])
        region = st.selectbox("Region", ["City", "Town", "Village"])
        membership = st.selectbox("Membership", ["Basic Membership", "No Membership", "Gold Membership", "Silver Membership", "Premium Membership", "Platinum Membership"])
        join_date = st.text_input("Joining Date (YYYY-MM-DD)", "2020-01-01")

    with c2:
        st.subheader("Engagement")
        time_spent = st.number_input("Time Spent (min)", 0.0, 1000.0, 200.0)
        trans_val = st.number_input("Transaction Value", 0.0, 100000.0, 30000.0)
        points = st.number_input("Wallet Points", 0.0, 2000.0, 450.0)
        last_login = st.number_input("Days Since Login", 0, 31, 7)

    with c3:
        st.subheader("Sentiment")
        feedback = st.selectbox("Review Text", review_options)
        complaint = st.selectbox("Past Complaint", ["Yes", "No"])
        internet = st.selectbox("Internet", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"])
        referral = st.selectbox("Referral?", ["Yes", "No"])

    btn = st.form_submit_button("Generate Prediction")

if btn:
    data = {
        "age": age, "gender": gender, "region_category": region, "membership_category": membership,
        "joining_date": join_date, "joined_through_referral": referral, "preferred_offer_types": "Without Offers",
        "medium_of_operation": "Desktop", "internet_option": internet, "last_visit_time": "14:30:00",
        "days_since_last_login": last_login, "avg_time_spent": time_spent, "avg_transaction_value": trans_val,
        "avg_frequency_login_days": 15.0, "points_in_wallet": points, "used_special_discount": "Yes",
        "offer_application_preference": "Yes", "past_complaint": complaint, "complaint_status": "Not Applicable",
        "feedback": feedback
    }
    
    input_df = pd.DataFrame([data])
    with st.spinner("Processing..."):
        try:
            result = pipeline.predict(input_df)
            score = result["score"] # KEY MATCH
            probs = result["probabilities"]
            
            st.divider()
            r1, r2 = st.columns(2)
            with r1:
                st.metric("Churn Risk Score", f"{score} / 5")
                if score >= 4: st.error("🚨 CRITICAL: High Churn Risk")
                elif score == 3: st.warning("⚖️ MODERATE: Retention Effort Needed")
                else: st.success("✅ STABLE: Low Churn Risk")
            
            with r2:
                st.write("Model Confidence Distribution")
                st.bar_chart(pd.DataFrame(probs, index=[1,2,3,4,5], columns=["Confidence"]))
        except Exception as e:
            st.error(f"Inference Failure: {e}")