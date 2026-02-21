import pandas as pd
import numpy as np

def generate_real_world_data(n=100000):
    np.random.seed(42)
    
    # --- 1. REALISTIC DEMOGRAPHICS (Skewed Distributions) ---
    # Income is never a Bell Curve; it's Log-Normal (Long tail of wealthy)
    income_dist = np.random.lognormal(mean=11.0, sigma=0.6, size=n)
    income = np.clip(income_dist, 25000, 2500000).round(-2)
    
    # Age: Bimodal distribution (Young professionals vs Retirees)
    age = np.concatenate([
        np.random.normal(30, 5, int(n*0.6)), # Young cohort
        np.random.normal(55, 8, int(n*0.4))  # Older cohort
    ]).astype(int)
    np.random.shuffle(age)
    
    # Sectors with different risk profiles
    sectors = np.random.choice(
        ['Tech', 'Healthcare', 'Real Estate', 'Retail', 'Energy', 'Hospitality'], 
        size=n, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    )
    
    # --- 2. BANKING BEHAVIORS (Churn Drivers) ---
    # Tenure: How long they've been a customer (0-10 years)
    tenure = np.random.randint(0, 11, n)
    
    # Number of Products: The "Sticky" factor
    # Real insight: 1 product = likely to leave. 2 = sticky. 3+ = problematic/high maintenance.
    num_products = np.random.choice([1, 2, 3, 4], size=n, p=[0.5, 0.4, 0.08, 0.02])
    
    # Is Active Member? (50% active)
    is_active = np.random.binomial(1, 0.51, n)
    
    # Balance (30% of people have 0 balance - highly realistic)
    balance = np.where(np.random.random(n) < 0.3, 0, np.random.normal(60000, 30000, n))
    balance = np.clip(balance, 0, None)

    # --- 3. CREDIT RISK MODELING (LendingClub Style) ---
    # Credit Score is correlated with Income and Age, but with noise
    base_score = 600 + (np.log(income)/15)*100 + (age/2) 
    noise = np.random.normal(0, 40, n)
    credit_score = np.clip(base_score + noise, 300, 850).astype(int)
    
    # Loan Amount depends on Income (DTI constraints)
    # But some people over-leverage (High Risk)
    leverage_factor = np.random.beta(2, 5, n) # Skewed towards lower DTI
    loan_amount = (income * leverage_factor * 3).clip(5000, 1000000).round(-2)
    
    dti = loan_amount / income
    
    # Interest Rate (Risk-Based Pricing + Market Volatility)
    # Lower score = Exponentially higher rate
    base_rate = 0.035 # Prime rate
    risk_premium = 0.25 * np.exp(-((credit_score - 300) / 150))
    interest_rate = (base_rate + risk_premium + np.random.normal(0, 0.005, n)).clip(0.04, 0.29)
    
    # --- 4. TARGET VARIABLES (The "Truth") ---
    
    # DEFAULT LOGIC (Complex)
    # Driven by: High DTI, Low Score, Low Income, Volatile Sectors
    sector_risk = np.array([1.5 if s in ['Retail', 'Hospitality'] else 1.0 for s in sectors])
    
    # Log-odds for default
    logit_default = -6 + (dti*3) - ((credit_score-600)/50) + (sector_risk*0.5) 
    prob_default = 1 / (1 + np.exp(-logit_default))
    default_history = np.random.binomial(1, prob_default)
    
    # CHURN LOGIC (Complex)
    # Driven by: High Rate, Low Tenure, 1 Product, Inactive
    # Real insight: Active members with 2 products rarely churn.
    logit_churn = -2 + (interest_rate*10) - (tenure*0.1) - (is_active*1.5)
    # Penalty for having only 1 product
    logit_churn += np.where(num_products == 1, 1.0, 0)
    
    prob_churn = 1 / (1 + np.exp(-logit_churn))
    churn_risk = np.random.binomial(1, prob_churn)
    
    # LGD (Loss Given Default)
    # Not fixed! Depends on loan size (Recoveries are harder on big loans)
    lgd = np.random.beta(2, 2, n).clip(0.1, 0.9) # Wide variance

    # --- 5. ASSEMBLE ---
    df = pd.DataFrame({
        'CustomerID': range(100001, 100001 + n),
        'Age': age,
        'Annual_Income': income,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Sector': sectors,
        'Tenure': tenure,
        'Num_Products': num_products,
        'Is_Active': is_active,
        'Account_Balance': balance.round(2),
        'LGD': lgd.round(2),
        'DTI_Ratio': dti.round(4),
        'Interest_Rate': interest_rate.round(4),
        'Default_History': default_history,
        'Churn_Risk': churn_risk
    })
    
    df.to_csv("alphawiz_data.csv", index=False)
    print(f"Generated {n} realistic banking records.")

if __name__ == "__main__":
    generate_real_world_data()