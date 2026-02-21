import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sqlalchemy import text

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="AlphaWiz Core", layout="wide", page_icon="🏦")
st.markdown("""
    <style>
        .block-container {padding-top: 1rem;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .big-font {font-size:20px !important;}
    </style>
""", unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_resource
def init_system():
    # Establishes connection to the SQL database (SQLite for local, Postgres for Prod)
    conn = st.connection("alphadb", type="sql")
    try:
        # Check if data exists by querying one row
        conn.query("SELECT 1 FROM loan_book LIMIT 1")
    except Exception:
        try:
            # If table missing, load from CSV and create table
            data = pd.read_csv("alphawiz_data.csv")
            with conn.session as s:
                data.to_sql("loan_book", conn.engine, if_exists="replace", index=False)
                s.commit()
        except FileNotFoundError:
             st.error("Critical Error: 'alphawiz_data.csv' is missing. Please run the 'generate_data.py' script first.")
             st.stop()
    return conn

conn = init_system()
# Load full dataset into memory for high-performance filtering
df = conn.query("SELECT * FROM loan_book")

# --- 2. INTELLIGENCE CORE (ML MODELS) ---
@st.cache_resource
def train_models(data):
    # Prepare Data for Risk Model (Handling Categorical 'Sector')
    data['Sector_Code'] = data['Sector'].astype('category').cat.codes
    
    # Feature Set 1: Credit Risk (PD Model)
    # Uses traditional banking variables + Sector risk
    X_risk = data[['Annual_Income', 'Loan_Amount', 'Credit_Score', 'DTI_Ratio', 'Sector_Code']]
    y_risk = data['Default_History']
    
    # Feature Set 2: Retention (Churn Model)
    # Uses behavioral "Sticky" variables (Tenure, Products, Activity)
    X_churn = data[['Interest_Rate', 'Credit_Score', 'Age', 'Tenure', 'Num_Products', 'Is_Active', 'Account_Balance']]
    y_churn = data['Churn_Risk']
    
    # Train Models (XGBoost)
    # Using max_depth=5 to capture non-linear real-world patterns
    risk_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_risk, y_risk)
    churn_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_churn, y_churn)
    
    return risk_model, churn_model, X_risk, X_churn

# Train or load models from cache
risk_model, churn_model, X_risk, X_churn = train_models(df)

# --- GLOBAL CALCULATIONS ---
# Calculated once here to be available across all dashboard tabs
total_aum = df['Loan_Amount'].sum()

# --- 3. DASHBOARD UI ---
st.sidebar.title("🏦 AlphaWiz Core")
st.sidebar.caption("Enterprise Risk & Intelligence Platform")
st.sidebar.markdown("---")

# Navigation
module = st.sidebar.radio("Select Module", 
    ["📊 Executive Overview", "✍️ Loan Origination", "📉 Stress Testing (Basel III)", "🧠 Retention Campaign Mgr"])

# ==============================================================================
# MODULE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if module == "📊 Executive Overview":
    st.title("Executive Portfolio Dashboard")
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    avg_roi = (df['Interest_Rate'].mean() * 100)
    risk_rate = (df['Default_History'].mean() * 100)
    
    kpi1.metric("Assets Under Management", f"€{total_aum/1e6:.1f}M")
    kpi2.metric("Avg. Interest Yield", f"{avg_roi:.2f}%")
    kpi3.metric("Portfolio Default Rate", f"{risk_rate:.2f}%")
    kpi4.metric("Active Customers", f"{len(df):,}")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Exposure by Sector")
        # Visualizing where our money is concentrated
        fig_sec = px.pie(df, names='Sector', values='Loan_Amount', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_sec, use_container_width=True)
    
    with col2:
        st.subheader("Credit Score Distribution")
        # Visualizing the quality of our borrowers
        fig_hist = px.histogram(df, x="Credit_Score", color="Sector", nbins=20, 
                                title="Risk Profile by Industry", labels={'Credit_Score': 'FICO Score'})
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# MODULE 2: LOAN ORIGINATION (Front Office)
# ==============================================================================
elif module == "✍️ Loan Origination":
    st.title("Smart Loan Origination System")
    st.markdown("Enter applicant details for real-time AI underwriting.")
    
    # Input Form
    with st.form("application_form"):
        c1, c2 = st.columns(2)
        with c1:
            app_income = st.number_input("Annual Income (€)", 20000, 1000000, 60000)
            app_loan = st.number_input("Requested Loan (€)", 5000, 500000, 150000)
            app_sector = st.selectbox("Industry Sector", df['Sector'].unique())
        with c2:
            app_score = st.slider("Credit Score (FICO)", 300, 850, 720)
            
        submitted = st.form_submit_button("Analyze Application")
        
    if submitted:
        # Feature Engineering for Single Prediction
        dti = app_loan / app_income
        # Map Sector string to the Code used in training
        # Note: In production, use a saved encoder. Here we map dynamically for the demo.
        sector_map = {sec: i for i, sec in enumerate(df['Sector'].astype('category').cat.categories)}
        sector_code = sector_map.get(app_sector, 0)
        
        input_data = pd.DataFrame([[app_income, app_loan, app_score, dti, sector_code]], 
                                  columns=['Annual_Income', 'Loan_Amount', 'Credit_Score', 'DTI_Ratio', 'Sector_Code'])
        
        # Prediction
        pd_prob = risk_model.predict_proba(input_data)[:, 1][0]
        
        # Decision Engine
        st.divider()
        c_res, c_explain = st.columns([1, 2])
        
        with c_res:
            if pd_prob < 0.15: # 15% PD Threshold
                st.success("✅ APPROVED")
                # Dynamic Pricing: Higher Risk = Higher Rate
                suggested_rate = 0.04 + (pd_prob * 0.5) 
                st.metric("Suggested Interest Rate", f"{suggested_rate:.2%}")
                st.metric("AI Risk Score (PD)", f"{pd_prob:.2%}")
            else:
                st.error("❌ REJECTED - High Risk")
                st.metric("Risk Score (PD)", f"{pd_prob:.2%}")
                st.caption("Auto-Rejection Threshold: PD > 15%")
                
        with c_explain:
            st.subheader("Why this decision? (Explainable AI)")
            explainer = shap.TreeExplainer(risk_model)
            shap_vals = explainer.shap_values(input_data)
            # Visualization
            plt.figure(figsize=(8, 3))
            shap.summary_plot(shap_vals, input_data, plot_type="bar", show=False, color='#FF4B4B')
            st.pyplot(plt.gcf())

# ==============================================================================
# MODULE 3: STRESS TESTING (Basel III Compliance)
# ==============================================================================
elif module == "📉 Stress Testing (Basel III)":
    st.title("🛡️ Capital Adequacy Monitor")
    st.markdown("""
    **Business Objective:** Ensure the bank holds enough capital to survive economic shocks.
    * **Regulatory Standard:** Basel III (Requires >10.5% Capital Ratio).
    * **Action:** If Ratio < 10.5%, the bank must freeze lending.
    """)
    
    # 1. Scenario Selection
    st.subheader("1. Select Macro-Economic Scenario")
    scenario = st.selectbox("Define Stress Level:", 
        ["Baseline (Normal Market)", 
         "Mild Recession (GDP -2%, Defaults +20%)", 
         "Severe Financial Crisis (GDP -5%, Defaults +50%)"])

    # 2. Apply Stress Logic
    if "Baseline" in scenario:
        shock_factor = 0.0
    elif "Mild" in scenario:
        shock_factor = 0.20
    else:
        shock_factor = 0.50

    # 3. Calculate Impact
    stressed_df = df.copy()
    # Stress the PD (Probability of Default)
    stressed_df['Base_PD'] = risk_model.predict_proba(X_risk)[:, 1]
    stressed_df['Stressed_PD'] = stressed_df['Base_PD'] * (1 + shock_factor)
    
    # Calculate Expected Loss (EL) = PD * LGD * EAD
    stressed_df['EL'] = stressed_df['Stressed_PD'] * stressed_df['LGD'] * stressed_df['Loan_Amount']
    
    total_el = stressed_df['EL'].sum()
    
    # Capital Calculations
    available_capital = total_aum * 0.12 # Assumption: We currently hold 12% reserves
    remaining_capital = available_capital - total_el
    capital_ratio = (remaining_capital / total_aum) * 100
    
    # 4. Results
    st.subheader("2. Regulatory Impact Analysis")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Projected Losses (EL)", f"€{total_el/1e6:.1f}M", 
              delta=f"Shock Impact: {shock_factor*100:.0f}%", delta_color="inverse")
    
    m2.metric("Remaining Capital Buffer", f"€{remaining_capital/1e6:.1f}M")
    
    m3.metric("Post-Stress Capital Ratio", f"{capital_ratio:.2f}%", 
              help="Must remain above 10.5% to pass audit")

    # 5. Pass/Fail Indicator
    st.divider()
    if capital_ratio > 10.5:
        st.success(f"✅ PASSED: Bank remains solvent under '{scenario}'. No action required.")
    else:
        st.error(f"🚨 FAILED: Capital drops below 10.5%. IMMEDIATE ACTION REQUIRED.")
        st.warning("**Recommended Actions:** 1. Halt lending in 'Real Estate' sector. 2. Raise €50M Tier 1 Capital.")

# ==============================================================================
# MODULE 4: RETENTION CAMPAIGN (Marketing ROI)
# ==============================================================================
elif module == "🧠 Retention Campaign Mgr":
    st.title("🚀 Customer Retention Engine")
    st.markdown("""
    **Business Objective:** Maximize ROI by preventing high-value customers from churning.
    * **Strategy:** Identify VIPs at risk and simulate the profitability of a 'Rate Discount' offer.
    """)
    
    # 1. Identify Target Segment
    df['Churn_Prob'] = churn_model.predict_proba(X_churn)[:, 1]
    
    # Filter: High Value (Score > 700) AND High Risk (Prob > 60%)
    vip_risk_segment = df[(df['Churn_Prob'] > 0.60) & (df['Credit_Score'] > 700)]
    
    total_risk_value = vip_risk_segment['Loan_Amount'].sum()
    st.info(f"🎯 **Target Segment:** {len(vip_risk_segment)} VIP Clients at risk. **Revenue at Risk:** €{total_risk_value/1e6:.1f}M")

    # 2. Campaign Simulator
    st.subheader("Campaign Simulator: 'Loyalty Rate Cut'")
    discount_offer = st.slider("Select Interest Rate Discount to Offer:", 0.0, 1.0, 0.2, step=0.1, format="-%.1f%%")
    
    # 3. ROI Calculation
    # Impact: Every 0.1% discount reduces churn probability by 10% (Assumption)
    impact_factor = (discount_offer * 10) * 0.10
    
    # Cost = Lost interest on the loan
    campaign_cost = total_risk_value * (discount_offer / 100)
    
    # Benefit = Value of loans saved from churning
    retained_loans_value = total_risk_value * impact_factor 
    
    roi = retained_loans_value - campaign_cost
    
    # 4. Actionable Outcomes
    c1, c2, c3 = st.columns(3)
    c1.metric("Campaign Cost (Lost Interest)", f"€{campaign_cost:,.0f}", delta="Upfront Cost", delta_color="inverse")
    c2.metric("Value of Retained Clients", f"€{retained_loans_value:,.0f}", delta="Long-term Value")
    c3.metric("Net Campaign ROI", f"€{roi:,.0f}", delta_color="normal" if roi > 0 else "inverse")

    st.subheader("Recommended Next Steps")
    if roi > 0:
        st.success(f"🚀 **LAUNCH CAMPAIGN:** This offer generates €{roi:,.0f} in net value.")
        # Show who to call
        st.write("Call List (Sample):")
        st.dataframe(
            vip_risk_segment[['CustomerID', 'Credit_Score', 'Tenure', 'Num_Products', 'Churn_Prob']].head(10),
            column_config={
                "Churn_Prob": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
                "Num_Products": st.column_config.NumberColumn("# Products", help="1 Product = High Churn Risk")
            }
        )
    else:
        st.error("🛑 **DO NOT LAUNCH:** The cost of the discount exceeds the value of customers saved. Try a lower discount.")