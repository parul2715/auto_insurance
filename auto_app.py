import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import streamlit as st
from itertools import combinations
from collections import Counter
import plotly.express as px
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

def set_dark_theme():
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] button {
            background-color: #999999 !important;
            color: #ffffff !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            border: none !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        section[data-testid="stSidebar"] button span {
            color: #ffffff !important;
            font-weight: bold !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .logout-container {
            margin-top: auto;
        }
        </style>
    """, unsafe_allow_html=True)



def plot_bar(data, x, y, title, xlabel, ylabel, hue=None, color=None, rotation=30, fmt="{:.1f}%"):
    plt.figure(figsize=(8, 6))
    sns.set_style("white")

    ax = sns.barplot(
        data=data, x=x, y=y, hue=hue,
        palette="pastel" if hue else None,
        color=color, dodge=False, legend=False
    )

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2., h),
                        ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=rotation)
    sns.despine()
    st.pyplot(plt.gcf())
    plt.clf()



def custom_kpi(label, value, delta=None, value_color="#ffffff", delta_color="#00ff99"):
    st.markdown(f"""
        <div style="padding: 15px 20px; border-radius: 12px; background-color: #1e1e1e;
                    color: white; border: 1px solid #333; margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: 500; color: #cccccc;">{label}</div>
            <div style="font-size: 24px; font-weight: bold; color: {value_color};">{value}</div>
            {f'<div style="font-size: 12px; color: {delta_color};">{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)

Users = {
    'Parul': '3456'
}

##--------------connection and loading data-----------------

@st.cache_data
def load_data():
    try:
        engine = create_engine(st.secrets["db_url"])
        df = pd.read_sql("SELECT * FROM insurance", engine)

        if "Effective_To_Date" in df.columns:
            df["Effective_To_Date"] = pd.to_datetime(df["Effective_To_Date"], errors="coerce")
            df["Month"] = df["Effective_To_Date"].dt.month_name()

        if "Response" in df.columns:
            df["Response"] = df["Response"].astype(str)

        if "CLV_Corrected" in df.columns:
            df["CLV_Segment"] = pd.qcut(df["CLV_Corrected"], 3, labels=["Low", "Medium", "High"])

        if "Income" in df.columns:
            df["Income_Bracket"] = pd.cut(
                df["Income"],
                bins=[-1, 25000, 50000, 75000, 100000, float("inf")],
                labels=["<25K", "25-50K", "50-75K", "75-100K", "100K+"]
            )

        if "Months_Since_Policy_Inception" in df.columns:
            df["Is_New"] = df["Months_Since_Policy_Inception"] < 12

        return df

    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return pd.DataFrame()
    

def login_page():
    set_dark_theme()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h4 style='text-align:center; color:#ccc;'>Welcome to the Auto Insurance Analysis Dashboard</h4>", unsafe_allow_html=True)
        st.title("🔐 Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if "login_attempted" not in st.session_state:
            st.session_state.login_attempted = False

        if st.button("Login", key="login_button"):
            st.session_state.login_attempted = True
            if username in Users and Users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "intro"
                st.success("Login Successful")
                st.rerun()

        if st.session_state.login_attempted and not st.session_state.logged_in:
            st.error("Invalid Username or Password", icon="🚫")



def business_context():
    set_dark_theme()
    st.markdown("""
        <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 90% !important;
        }
        div.stButton > button {
            background-color: #0099FF;
            color: white;
            font-size: 16px;
            padding: 10px 30px;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
            display: inline-flex;
            white-space: nowrap;
        }
        div.stButton > button:hover {
            background-color: #007ACC;
        }
        </style>
    """, unsafe_allow_html=True)
    img_col1, img_col2, img_col3 = st.columns([2, 3, 1])
    with img_col2:
        st.image("image.webp", width=300, use_container_width=True)
    st.markdown("<h1 style='text-align:center; font-size:36px;'>💼 Business Context</h1>", unsafe_allow_html=True)
    text_col1, text_col2, text_col3 = st.columns([1, 6, 1])
    with text_col2:
        st.markdown("""
            <p style='text-align:justify; font-size:18px; color:#CCCCCC;'>
                The auto insurance industry faces challenges in risk assessment, fraud detection, and customer retention. This project uses data-driven analysis and 
                predictive modeling to help insurers understand customer behavior, predict claims, and optimize campaigns, leading to better pricing, reduced losses, 
                and improved customer satisfaction.
            </p>
        """, unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([4, 2, 4])
    with btn_col2:
        if st.button("👉 Continue to Dashboard", key="continue_to_dashboard_btn"):
            st.session_state.page = "dashboard"
            st.rerun()

##---------------overall business snapshot-----------------------

def overall_business_snapshot(df):
    st.header("Overall Business Snapshot")
    months = ["All"] + sorted(df['Month'].dropna().unique())
    selected_month = st.selectbox("Select Month", months)
    if selected_month != "All":
        df = df[df['Month'] == selected_month]
    total_customers = df['Customer'].nunique()
    custom_kpi("Total Customers", f"{total_customers:,}")
    total_premium_revenue = df['Monthly_Premium_Auto'].sum()
    custom_kpi("Total Premium Revenue", f"${total_premium_revenue:,.2f}")
    avg_premium_per_customer = df['Monthly_Premium_Auto'].mean()
    custom_kpi("Avg Premium per Customer", f"${avg_premium_per_customer:,.2f}")
    total_claims = df['Total_Claim_Amount'].sum()
    claim_rate = (df['Total_Claim_Amount'] > 0).mean()
    custom_kpi("Total Claims Amount", f"${total_claims:,.2f}")
    custom_kpi("Claim Rate", f"{claim_rate:.2%}")
    campaign_response_rate = (df['Response'] == 1).mean()
    custom_kpi("Campaign Response Rate", f"{campaign_response_rate:.2%}")
    avg_clv = df['CLV_Corrected'].mean()
    custom_kpi("Average CLV", f"${avg_clv:,.2f}")

##----------------------customer demographics analysis----------------------------------------

def customer_demographics(df_filtered):
    st.header("Customer Demographics Analysis")

    # Customer Distribution by State
    state_dist = df_filtered['State'].value_counts().reset_index()
    state_dist.columns = ['State', 'Customer_Count']
    plot_bar(
        state_dist,
        x="State",
        y="Customer_Count",
        title="Customer Distribution by State",
        xlabel="State",
        ylabel="Number of Customers",
        hue="State",
        rotation=45,
        fmt="{:.0f}"
    )

    # Gender Distribution
    gender_dist = df_filtered['Gender'].value_counts().reset_index()
    gender_dist.columns = ['Gender', 'Count']
    plot_bar(
        gender_dist,
        x="Gender",
        y="Count",
        title="Gender Distribution",
        xlabel="Gender",
        ylabel="Count",
        hue="Gender",
        fmt="{:.0f}"
    )

    # Marital Status Distribution
    marital_dist = df_filtered['Marital_Status'].value_counts().reset_index()
    marital_dist.columns = ['Marital_Status', 'Count']
    plot_bar(
        marital_dist,
        x="Marital_Status",
        y="Count",
        title="Marital Status Distribution",
        xlabel="Marital Status",
        ylabel="Count",
        hue="Marital_Status",
        fmt="{:.0f}"
    )

    # Education Level Breakdown
    edu_dist = df_filtered['Education'].value_counts().reset_index()
    edu_dist.columns = ['Education', 'Count']
    plot_bar(
        edu_dist,
        x="Education",
        y="Count",
        title="Education Level Breakdown",
        xlabel="Education Level",
        ylabel="Count",
        hue="Education",
        rotation=30,
        fmt="{:.0f}"
    )

    # Income Distribution – KEEP as histplot (not bar)
    plt.figure(figsize=(8,6))
    ax = sns.histplot(
        data=df_filtered,
        x="Income",
        bins=30,
        kde=True,
        color="teal"
    )
    plt.title("Income Distribution", fontsize=14, weight='bold')
    plt.xlabel("Annual Income", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    st.pyplot(plt.gcf())
    plt.clf()

    # Employment Status
    emp_dist = df_filtered['EmploymentStatus'].value_counts().reset_index()
    emp_dist.columns = ['EmploymentStatus', 'Count']
    plot_bar(
        emp_dist,
        x="EmploymentStatus",
        y="Count",
        title="Job Role Distribution",
        xlabel="Employment Status",
        ylabel="Count",
        hue="EmploymentStatus",
        rotation=30,
        fmt="{:.0f}"
    )

    # Vehicle Class
    veh_dist = df_filtered['Vehicle_Class'].value_counts().reset_index()
    veh_dist.columns = ['Vehicle_Class', 'Count']
    plot_bar(
        veh_dist,
        x="Vehicle_Class",
        y="Count",
        title="Vehicle Class Distribution",
        xlabel="Vehicle Class",
        ylabel="Count",
        hue="Vehicle_Class",
        rotation=30,
        fmt="{:.0f}"
    )

    # Avg Income by Education
    income_by_edu = df_filtered.groupby('Education')['Income'].mean().reset_index().sort_values(by='Income', ascending=False)
    plot_bar(
        income_by_edu,
        x="Education",
        y="Income",
        title="Average Income by Education Level",
        xlabel="Education Level",
        ylabel="Average Income (USD)",
        hue="Education",
        rotation=30,
        fmt="{:.0f}"
    )

    # Avg CLV by Education
    clv_by_edu = df_filtered.groupby('Education')['CLV_Corrected'].mean().reset_index().sort_values(by='CLV_Corrected', ascending=False)
    plot_bar(
        clv_by_edu,
        x="Education",
        y="CLV_Corrected",
        title="Average CLV by Education Level",
        xlabel="Education",
        ylabel="Avg CLV (Corrected)",
        hue="Education",
        rotation=30,
        fmt="{:.0f}"
    )

    # Avg CLV by Gender
    clv_by_gender = df_filtered.groupby('Gender')['CLV_Corrected'].mean().reset_index().sort_values(by='CLV_Corrected', ascending=False)
    plot_bar(
        clv_by_gender,
        x="Gender",
        y="CLV_Corrected",
        title="Average CLV by Gender",
        xlabel="Gender",
        ylabel="Avg CLV (Corrected)",
        hue="Gender",
        fmt="{:.0f}"
    )

    # Avg CLV by State
    clv_by_state = df_filtered.groupby('State')['CLV_Corrected'].mean().reset_index().sort_values(by='CLV_Corrected', ascending=False)
    plot_bar(
        clv_by_state,
        x="State",
        y="CLV_Corrected",
        title="Average CLV by State",
        xlabel="State",
        ylabel="Avg CLV (Corrected)",
        hue="State",
        rotation=45,
        fmt="{:.0f}"
    )

    # Policies by State
    policies_state = df_filtered.groupby('State')['Number_of_Policies'].sum().reset_index().sort_values(by='Number_of_Policies', ascending=False)
    plot_bar(
        policies_state,
        x="State",
        y="Number_of_Policies",
        title="Total Number of Policies by State",
        xlabel="State",
        ylabel="Total Number of Policies",
        hue="State",
        rotation=45,
        fmt="{:.0f}"
    )

    # Policies by Gender
    policies_gender = df_filtered.groupby('Gender')['Number_of_Policies'].sum().reset_index().sort_values(by='Number_of_Policies', ascending=False)
    plot_bar(
        policies_gender,
        x="Gender",
        y="Number_of_Policies",
        title="Total Number of Policies by Gender",
        xlabel="Gender",
        ylabel="Total Number of Policies",
        hue="Gender",
        fmt="{:.0f}"
    )

    # Policies by Education
    policies_edu = df_filtered.groupby('Education')['Number_of_Policies'].sum().reset_index().sort_values(by='Number_of_Policies', ascending=False)
    plot_bar(
        policies_edu,
        x="Education",
        y="Number_of_Policies",
        title="Total Number of Policies by Education",
        xlabel="Education Level",
        ylabel="Total Number of Policies",
        hue="Education",
        rotation=30,
        fmt="{:.0f}"
    )

##------------------policy and coverage analysis----------------

def policy_coverage(df_filtered):
    st.header("Policy and Coverage Analysis")

    # Policy Type Distribution
    policy_type_dist = df_filtered['Policy_Type'].value_counts(normalize=True).reset_index()
    policy_type_dist.columns = ['Policy_Type', 'Percentage']
    policy_type_dist['Percentage'] *= 100
    plot_bar(
        policy_type_dist,
        x="Policy_Type",
        y="Percentage",
        title="Policy Type Distribution",
        xlabel="Policy Type",
        ylabel="Percentage (%)",
        hue="Policy_Type",
        rotation=30,
        fmt="{:.1f}%"
    )

    # Coverage Distribution
    coverage_dist = df_filtered['Coverage'].value_counts(normalize=True).reset_index()
    coverage_dist.columns = ['Coverage', 'Percentage']
    coverage_dist['Percentage'] *= 100
    plot_bar(
        coverage_dist,
        x="Coverage",
        y="Percentage",
        title="Coverage Type Distribution",
        xlabel="Coverage Type",
        ylabel="Percentage (%)",
        hue="Coverage",
        fmt="{:.1f}%"
    )

    # Tenure Bucket Distribution
    tenure_bins = [0, 12, 36, 120]
    tenure_labels = ['<1yr', '1-3yrs', '3+yrs']
    df_filtered['Tenure_Bucket'] = pd.cut(df_filtered['Months_Since_Policy_Inception'], bins=tenure_bins, labels=tenure_labels, right=False)
    tenure_bucket_dist = df_filtered['Tenure_Bucket'].value_counts(normalize=True).reset_index()
    tenure_bucket_dist.columns = ['Tenure_Bucket', 'Percentage']
    tenure_bucket_dist['Percentage'] *= 100
    plot_bar(
        tenure_bucket_dist,
        x="Tenure_Bucket",
        y="Percentage",
        title="Policy Tenure Buckets Distribution",
        xlabel="Tenure Bucket",
        ylabel="Percentage (%)",
        hue="Tenure_Bucket",
        fmt="{:.1f}%"
    )

    # Avg Tenure
    avg_tenure = df_filtered['Months_Since_Policy_Inception'].mean()
    st.markdown(f"**Average Policy Tenure (Months):** {avg_tenure:.2f}")

    # Cross-analysis: KEEP as stacked barplot (not compatible with plot_bar)
    # Segmentation: KEEP as grouped barplot

    # Avg Premium by Coverage
    avg_premium = df_filtered.groupby('Coverage')['Monthly_Premium_Auto'].mean().reset_index().sort_values(by='Monthly_Premium_Auto', ascending=False)
    plot_bar(
        avg_premium,
        x="Coverage",
        y="Monthly_Premium_Auto",
        title="Average Premium by Coverage Level",
        xlabel="Coverage Type",
        ylabel="Average Monthly Premium",
        hue="Coverage",
        fmt="{:.0f}"
    )

    # Premium Contribution by Policy Type
    contrib = df_filtered.groupby('Policy_Type')['Monthly_Premium_Auto'].sum().reset_index().sort_values(by='Monthly_Premium_Auto', ascending=False)
    plot_bar(
        contrib,
        x="Policy_Type",
        y="Monthly_Premium_Auto",
        title="Premium Contribution by Policy Type",
        xlabel="Policy Type",
        ylabel="Total Premium ($)",
        hue="Policy_Type",
        fmt="{:.0f}"
    )

def claims_risk(df_filtered):
    st.header("⚠️ Claims and Risk Analysis")

    if df_filtered.empty:
        st.warning("No data available for selected filters.")
        return

    # ---------------- Key Claims Metrics ----------------
    total_claims = df_filtered['Number_of_Open_Complaints'].sum()
    claim_rate = (df_filtered[df_filtered['Number_of_Open_Complaints'] > 0].shape[0] / df_filtered.shape[0]) * 100
    avg_claims = df_filtered['Number_of_Open_Complaints'].mean()
    total_claim_amount = df_filtered['Total_Claim_Amount'].sum()
    avg_claim_amount = df_filtered['Total_Claim_Amount'].mean()
    loss_ratio = (total_claim_amount / df_filtered['Monthly_Premium_Auto'].sum()) * 100

    st.markdown(f"**Total Claims Count:** {total_claims}")
    st.markdown(f"**Claim Rate (%):** {claim_rate:.2f}")
    st.markdown(f"**Average Claims per Customer:** {avg_claims:.2f}")
    st.markdown(f"**Total Claim Amount:** ${total_claim_amount:.2f}")
    st.markdown(f"**Average Claim Amount:** ${avg_claim_amount:.2f}")
    st.markdown(f"**Loss Ratio (%):** {loss_ratio:.2f}")

    # ---------------- Claims by Policy Type ----------------
    claims_by_policy = (
        df_filtered.groupby('Policy_Type')['Total_Claim_Amount']
        .sum()
        .reset_index()
        .sort_values(by='Total_Claim_Amount', ascending=False)
    )
    plot_bar(
        claims_by_policy,
        x="Policy_Type",
        y="Total_Claim_Amount",
        title="Total Claims by Policy Type",
        xlabel="Policy Type",
        ylabel="Total Claim Amount",
        hue="Policy_Type",
        fmt="{:.0f}"
    )

    # ---------------- Claims by Vehicle Class ----------------
    claims_by_vehicle = (
        df_filtered.groupby('Vehicle_Class')['Total_Claim_Amount']
        .sum()
        .reset_index()
        .sort_values(by='Total_Claim_Amount', ascending=False)
    )
    plot_bar(
        claims_by_vehicle,
        x="Vehicle_Class",
        y="Total_Claim_Amount",
        title="Total Claims by Vehicle Class",
        xlabel="Vehicle Class",
        ylabel="Total Claim Amount",
        hue="Vehicle_Class",
        fmt="{:.0f}"
    )

    # ---------------- Claim Rate by Coverage Level ----------------
    claim_rate_coverage = (
        df_filtered.groupby('Coverage', observed=True).apply(
            lambda x: (x['Number_of_Open_Complaints'] > 0).sum() / x.shape[0] * 100,
            include_groups=False
        ).reset_index(name='Claim_Rate(%)')
    )
    claim_rate_coverage = claim_rate_coverage.sort_values(by='Claim_Rate(%)', ascending=False)

    plot_bar(
        claim_rate_coverage,
        x="Coverage",
        y="Claim_Rate(%)",
        title="Claim Rate by Coverage Level",
        xlabel="Coverage",
        ylabel="Claim Rate (%)",
        hue="Coverage",
        fmt="{:.1f}%"
    )

    # ---------------- Risk Segment Distribution ----------------
    df_filtered = df_filtered.copy()  # avoid SettingWithCopyWarning
    df_filtered['Risk_Segment'] = np.where(df_filtered['Number_of_Open_Complaints'] > 2, 'High', 'Normal')

    risk_segment_dist = (
        df_filtered['Risk_Segment']
        .value_counts(normalize=True)
        .reset_index()
        .rename(columns={'index': 'Risk_Segment', 'Risk_Segment': 'Percentage'})
    )
    risk_segment_dist['Percentage'] *= 100
    risk_segment_dist = risk_segment_dist.sort_values(by='Percentage', ascending=False)

    plot_bar(
        risk_segment_dist,
        x="Risk_Segment",
        y="Percentage",
        title="Risk Segment Distribution",
        xlabel="Risk Segment",
        ylabel="Percentage (%)",
        hue="Risk_Segment",
        rotation=0,
        fmt="{:.1f}%"
    )
##------------------------------clv analysis-----------------------------

def customer_ltv(df_filtered):
    st.header("💡 Customer Lifetime Value (CLV) Analysis")

    if df_filtered.empty:
        st.warning("No data available for selected filters.")
        return

    # ---------------- Key CLV Metrics ----------------
    avg_clv = df_filtered['CLV_Corrected'].mean()
    median_clv = df_filtered['CLV_Corrected'].median()
    total_clv = df_filtered['CLV_Corrected'].sum()
    high_clv_pct = (
        df_filtered[df_filtered['CLV_Corrected'] > df_filtered['CLV_Corrected'].quantile(0.75)].shape[0]
        / df_filtered.shape[0]
    ) * 100
    clv_premium_corr = df_filtered['CLV_Corrected'].corr(df_filtered['Monthly_Premium_Auto'])

    st.markdown(f"**Average CLV:** ${avg_clv:.2f}")
    st.markdown(f"**Median CLV:** ${median_clv:.2f}")
    st.markdown(f"**Total CLV:** ${total_clv:.2f}")
    st.markdown(f"**High CLV Customers (%):** {high_clv_pct:.2f}")
    st.markdown(f"**Correlation (CLV vs. Premium):** {clv_premium_corr:.2f}")

    # ---------------- CLV by Policy Type ----------------
    clv_by_policy = (
        df_filtered.groupby('Policy_Type')['CLV_Corrected']
        .mean()
        .reset_index()
        .sort_values(by='CLV_Corrected', ascending=False)
    )
    plot_bar(
        clv_by_policy,
        x="Policy_Type",
        y="CLV_Corrected",
        title="Average CLV by Policy Type",
        xlabel="Policy Type",
        ylabel="Average CLV",
        hue="Policy_Type"
    )

    # ---------------- CLV by Coverage Level ----------------
    clv_by_coverage = (
        df_filtered.groupby('Coverage')['CLV_Corrected']
        .mean()
        .reset_index()
        .sort_values(by='CLV_Corrected', ascending=False)
    )
    plot_bar(
        clv_by_coverage,
        x="Coverage",
        y="CLV_Corrected",
        title="Average CLV by Coverage Level",
        xlabel="Coverage",
        ylabel="Average CLV",
        hue="Coverage"
    )

    # ---------------- CLV Segment Distribution ----------------
    clv_segment_dist = (
        df_filtered['CLV_Segment'].value_counts(normalize=True)
        .reset_index()
        .rename(columns={'index': 'CLV_Segment', 'CLV_Segment': 'Percentage'})
    )
    clv_segment_dist['Percentage'] *= 100
    clv_segment_dist = clv_segment_dist.sort_values(by='Percentage', ascending=False)

    plot_bar(
        clv_segment_dist,
        x="CLV_Segment",
        y="Percentage",
        title="CLV Segment Distribution",
        xlabel="CLV Segment",
        ylabel="Percentage (%)",
        hue="CLV_Segment",
        rotation=0,
        fmt="{:.1f}%"
    )

    # ---------------- CLV by Sales Channel ----------------
    clv_by_channel = (
        df_filtered.groupby('Sales_Channel')['CLV_Corrected']
        .mean()
        .reset_index()
        .sort_values(by='CLV_Corrected', ascending=False)
    )
    plot_bar(
        clv_by_channel,
        x="Sales_Channel",
        y="CLV_Corrected",
        title="Average CLV by Sales Channel",
        xlabel="Sales Channel",
        ylabel="Average CLV",
        hue="Sales_Channel"
    )

    # ---------------- CLV by Retention Offer ----------------
    clv_retention = (
        df_filtered.groupby('Renew_Offer_Type', observed=True)['CLV_Corrected']
        .mean()
        .reset_index()
        .sort_values(by='CLV_Corrected', ascending=False)
    )
    plot_bar(
        clv_retention,
        x="Renew_Offer_Type",
        y="CLV_Corrected",
        title="Average CLV by Retention Offer",
        xlabel="Renew Offer Type",
        ylabel="Average CLV",
        hue="Renew_Offer_Type"
    )

    # ---------------- Growth Potential (25%-75%) ----------------
    lower_bound = df_filtered['CLV_Corrected'].quantile(0.25)
    upper_bound = df_filtered['CLV_Corrected'].quantile(0.75)
    growth_potential_customers = df_filtered[
        (df_filtered['CLV_Corrected'] > lower_bound) &
        (df_filtered['CLV_Corrected'] < upper_bound)
    ]
    st.markdown(f"**Number of Growth Potential Customers (25%-75% segment):** {len(growth_potential_customers)}")

    # ---------------- Top 10% Contribution ----------------
    top_10pct_count = int(0.1 * df_filtered.shape[0])
    top_10pct_clv = df_filtered.nlargest(top_10pct_count, 'CLV_Corrected')['CLV_Corrected'].sum()
    top_10pct_clv_share = (top_10pct_clv / df_filtered['CLV_Corrected'].sum()) * 100
    st.markdown(f"**Top 10% CLV Contribution (%):** {top_10pct_clv_share:.2f}")



##-------------MARKETING AND CAMPAIGN PERFORMANCE ANALYSIS



def marketing_campaigns(df_filtered):
    st.header("📢 Marketing and Campaign Performance Analysis")

    if df_filtered.empty:
        st.warning("No data available for selected filters.")
        return

    # ---------------- Response Rate by Channel ----------------
    response_by_channel = df_filtered.groupby('Sales_Channel').apply(
        lambda x: (x['Response'] == 'True').mean() * 100,
        include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plot_bar(
        response_by_channel,
        x="Sales_Channel",
        y="Response_Rate(%)",
        title="Response Rate by Sales Channel",
        xlabel="Sales Channel",
        ylabel="Response Rate (%)",
        hue="Sales_Channel"
    )

    # ---------------- Response Rate by Policy Type ----------------
    response_by_policy = df_filtered.groupby('Policy_Type').apply(
        lambda x: (x['Response'] == 'True').mean() * 100,
        include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plot_bar(
        response_by_policy,
        x="Policy_Type",
        y="Response_Rate(%)",
        title="Response Rate by Policy Type",
        xlabel="Policy Type",
        ylabel="Response Rate (%)",
        color="lightgreen"
    )

    # ---------------- Response Rate by Coverage ----------------
    response_by_coverage = df_filtered.groupby('Coverage').apply(
        lambda x: (x['Response'] == 'True').mean() * 100,
        include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plot_bar(
        response_by_coverage,
        x="Coverage",
        y="Response_Rate(%)",
        title="Response Rate by Coverage",
        xlabel="Coverage",
        ylabel="Response Rate (%)",
        color="lightcoral"
    )

    # ---------------- Response Rate by CLV Segment ----------------
    response_by_clv = df_filtered.groupby('CLV_Segment', observed=True).apply(
        lambda x: (x['Response'] == 'True').mean() * 100,
        include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plot_bar(
        response_by_clv,
        x="CLV_Segment",
        y="Response_Rate(%)",
        title="Response Rate by CLV Segment",
        xlabel="CLV Segment",
        ylabel="Response Rate (%)",
        color="lightskyblue",
        rotation=0
    )

    # ---------------- Offer Type Performance ----------------
    offers = ['Offer1', 'Offer2', 'Offer3', 'Offer4']
    response_by_offer = df_filtered.groupby('Renew_Offer_Type').apply(
        lambda x: (x['Response'] == 'True').mean() * 100,
        include_groups=False
    ).reset_index(name='Response_Rate(%)')
    response_by_offer = response_by_offer.set_index('Renew_Offer_Type').reindex(offers, fill_value=0).reset_index()

    plot_bar(
        response_by_offer,
        x="Renew_Offer_Type",
        y="Response_Rate(%)",
        title="Response Rate by Offer Type",
        xlabel="Renew Offer Type",
        ylabel="Response Rate (%)",
        color="lightpink"
    )

    # ---------------- Income Bracket Response Rate ----------------
    income_response = df_filtered.groupby('Income_Bracket', observed=True).apply(
        lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plot_bar(
        income_response,
        x="Income_Bracket",
        y="Response_Rate(%)",
        title="Response Rate by Income Bracket",
        xlabel="Income Bracket",
        ylabel="Response Rate (%)",
        color="lightblue"
    )

    # ---------------- Top Converting Segment (Policy Type x Coverage) ----------------
    top_segment = df_filtered.groupby(['Policy_Type', 'Coverage'], observed=True).apply(
        lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

    plt.figure(figsize=(8, 7))
    top_segment_sorted = top_segment.sort_values(by='Response_Rate(%)', ascending=False)
    ax = sns.barplot(data=top_segment_sorted, x='Coverage', y='Response_Rate(%)',
                     hue='Policy_Type', palette='pastel')
    plt.title('Top Converting Segments: Response Rate by Policy Type and Coverage')
    plt.xlabel('Coverage Level')
    plt.ylabel('Response Rate (%)')
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width()/2., h),
                        ha='center', va='bottom', fontsize=11, color='black')
    plt.legend(title='Policy Type')
    plt.xticks(rotation=30)
    sns.despine()
    st.pyplot(plt.gcf())
    plt.clf()

    # ---------------- Retention vs. New Acquisition Response ----------------
    retention_response = df_filtered.groupby('Is_New', observed=True).apply(
        lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
    ).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)
    retention_response['Customer_Type'] = retention_response['Is_New'].map({True: 'New', False: 'Retention'})

    plot_bar(
        retention_response,
        x="Customer_Type",
        y="Response_Rate(%)",
        title="Response Rate: Retention vs. New Acquisition",
        xlabel="Customer Type",
        ylabel="Response Rate (%)",
        color="lightcoral"
    )






def dashboard():
    set_dark_theme()
    st.sidebar.title("Navigation")
    pages = [
        "Overall Business Snapshot",
        "Customer Demographics Analysis",
        "Policy and Coverage Analysis",
        "Claims and Risk Analysis",
        "Customer Lifetime Value Analysis",
        "Marketing Campaign Performance Analysis"
    ]

    # Store selected page in session state to access outside sidebar
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = pages[0]
    st.session_state.selected_page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.selected_page))

    # Prepare filter options
    months = sorted(df_insurance['Month'].dropna().unique())
    states = sorted(df_insurance['State'].dropna().unique())
    employment_statuses = sorted(df_insurance['EmploymentStatus'].dropna().unique())
    sales_channels = sorted(df_insurance['Sales_Channel'].dropna().unique())
    coverages = sorted(df_insurance['Coverage'].dropna().unique())
    educations = sorted(df_insurance['Education'].dropna().unique())
    policy_types = sorted(df_insurance['Policy_Type'].dropna().unique())

    # Inside the sidebar, show filters depending on page and store selections in session state
    with st.sidebar:
        if st.session_state.selected_page == "Customer Demographics Analysis":
            st.session_state.month_filter = st.multiselect("Month", months, default=st.session_state.get("month_filter", months))
            st.session_state.state_filter = st.multiselect("State", states, default=st.session_state.get("state_filter", states))
            st.session_state.employment_filter = st.multiselect("Employment Status", employment_statuses, default=st.session_state.get("employment_filter", employment_statuses))

        elif st.session_state.selected_page == "Policy and Coverage Analysis":
            st.session_state.month_filter = st.multiselect("Month", months, default=st.session_state.get("month_filter", months))
            st.session_state.state_filter = st.multiselect("State", states, default=st.session_state.get("state_filter", states))
            st.session_state.sales_channel_filter = st.multiselect("Sales Channel", sales_channels, default=st.session_state.get("sales_channel_filter", sales_channels))

        elif st.session_state.selected_page == "Claims and Risk Analysis":
            st.session_state.month_filter = st.multiselect("Month", months, default=st.session_state.get("month_filter", months))
            st.session_state.employment_filter = st.multiselect("Employment Status", employment_statuses, default=st.session_state.get("employment_filter", employment_statuses))
            st.session_state.coverage_filter = st.multiselect("Coverage", coverages, default=st.session_state.get("coverage_filter", coverages))
            st.session_state.education_filter = st.multiselect("Education", educations, default=st.session_state.get("education_filter", educations))

        elif st.session_state.selected_page == "Customer Lifetime Value Analysis":
            st.session_state.month_filter = st.multiselect("Month", months, default=st.session_state.get("month_filter", months))
            st.session_state.state_filter = st.multiselect("State", states, default=st.session_state.get("state_filter", states))
            st.session_state.coverage_filter = st.multiselect("Coverage", coverages, default=st.session_state.get("coverage_filter", coverages))
            st.session_state.policy_type_filter = st.multiselect("Policy Type", policy_types, default=st.session_state.get("policy_type_filter", policy_types))

        elif st.session_state.selected_page == "Marketing Campaign Performance Analysis":
            st.session_state.month_filter = st.multiselect("Month", months, default=st.session_state.get("month_filter", months))
            st.session_state.policy_type_filter = st.multiselect("Policy Type", policy_types, default=st.session_state.get("policy_type_filter", policy_types))
            st.session_state.sales_channel_filter = st.multiselect("Sales Channel", sales_channels, default=st.session_state.get("sales_channel_filter", sales_channels))

        if st.session_state.get("logged_in", False):
            st.markdown("---")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.experimental_rerun()

    # Render page content outside sidebar using session state filters
    selected_page = st.session_state.selected_page

    if selected_page == "Overall Business Snapshot":
        overall_business_snapshot(df_insurance)

    elif selected_page == "Customer Demographics Analysis":
        df_filtered = df_insurance[
            df_insurance['Month'].isin(st.session_state.get("month_filter", months)) &
            df_insurance['State'].isin(st.session_state.get("state_filter", states)) &
            df_insurance['EmploymentStatus'].isin(st.session_state.get("employment_filter", employment_statuses))
        ]
        customer_demographics(df_filtered)

    elif selected_page == "Policy and Coverage Analysis":
        df_filtered = df_insurance[
            df_insurance['Month'].isin(st.session_state.get("month_filter", months)) &
            df_insurance['State'].isin(st.session_state.get("state_filter", states)) &
            df_insurance['Sales_Channel'].isin(st.session_state.get("sales_channel_filter", sales_channels))
        ]
        policy_coverage(df_filtered)

    elif selected_page == "Claims and Risk Analysis":
        df_filtered = df_insurance[
            df_insurance['Month'].isin(st.session_state.get("month_filter", months)) &
            df_insurance['EmploymentStatus'].isin(st.session_state.get("employment_filter", employment_statuses)) &
            df_insurance['Coverage'].isin(st.session_state.get("coverage_filter", coverages)) &
            df_insurance['Education'].isin(st.session_state.get("education_filter", educations))
        ]
        claims_risk(df_filtered)

    elif selected_page == "Customer Lifetime Value Analysis":
        df_filtered = df_insurance[
            df_insurance['Month'].isin(st.session_state.get("month_filter", months)) &
            df_insurance['State'].isin(st.session_state.get("state_filter", states)) &
            df_insurance['Coverage'].isin(st.session_state.get("coverage_filter", coverages)) &
            df_insurance['Policy_Type'].isin(st.session_state.get("policy_type_filter", policy_types))
        ]
        customer_ltv(df_filtered)

    elif selected_page == "Marketing Campaign Performance Analysis":
        df_filtered = df_insurance[
            df_insurance['Month'].isin(st.session_state.get("month_filter", months)) &
            df_insurance['Policy_Type'].isin(st.session_state.get("policy_type_filter", policy_types)) &
            df_insurance['Sales_Channel'].isin(st.session_state.get("sales_channel_filter", sales_channels))
        ]
        marketing_campaigns(df_filtered)



##-----main app control

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if st.session_state.logged_in:
    
    if st.session_state.page == "business":
        business_context()
    elif st.session_state.page == "dashboard":
        dashboard()
    else:
        dashboard()
else:
    login_page()