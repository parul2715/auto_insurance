import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')
import pyodbc
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

# SQL connection and data load
server_name = "LAPTOP-1FURGFHN\\SQLEXPRESS"
database_name = "auto_insurance"
driver = "ODBC+Driver+17+for+SQL+Server"
connection_string = f"mssql+pyodbc://{server_name}/{database_name}?driver={driver}&trusted_connection=yes"
engine = create_engine(connection_string)
query = "SELECT * FROM insurance"
df_insurance = pd.read_sql(query, engine)
df_insurance['Effective_To_Date'] = pd.to_datetime(df_insurance['Effective_To_Date'], errors='coerce')
df_insurance['Month'] = df_insurance['Effective_To_Date'].dt.month_name()

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

def customer_demographics(df_filtered):
    st.header("Customer Demographics Analysis")
    
    # Customer Distribution by State
    state_counts = df_filtered['State'].value_counts()
    plt.figure(figsize=(10,6))
    ax = sns.barplot(
        x=state_counts.index,
        y=state_counts.values,
        hue=state_counts.index,
        palette="viridis",
        legend=False
    )
    plt.title("Customer Distribution by State", fontsize=14, weight='bold')
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Number of Customers", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    # Gender Distribution
    plt.figure(figsize=(6,6))
    sns.set_style("white")
    ax = sns.countplot(
        data=df_filtered,
        x="Gender",
        hue="Gender",
        palette="Set2",
        legend=False
    )
    plt.title("Gender Distribution", fontsize=14, weight='bold')
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    st.pyplot(plt.gcf())
    plt.clf()

    # Marital Status Distribution
    plt.figure(figsize=(6,6))
    sns.set_style("white")
    ax = sns.countplot(
        data=df_filtered,
        x="Marital_Status",
        hue="Marital_Status",
        palette="pastel",
        legend=False
    )
    plt.title("Marital Status Distribution", fontsize=14, weight='bold')
    plt.xlabel("Marital Status", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    st.pyplot(plt.gcf())
    plt.clf()

    # Education Level Breakdown
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.countplot(
        data=df_filtered,
        x="Education",
        hue="Education",
        palette="coolwarm",
        legend=False
    )
    plt.title("Education Level Breakdown", fontsize=14, weight='bold')
    plt.xlabel("Education Level", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', 
                        (p.get_x() + p.get_width()/2., height), 
                        ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30)
    st.pyplot(plt.gcf())
    plt.clf()

    # Income Distribution
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
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8)
    st.pyplot(plt.gcf())
    plt.clf()

    # Job Role Distribution
    plt.figure(figsize=(10,6))
    ax = sns.countplot(
        data=df_filtered,
        x="EmploymentStatus",
        hue="EmploymentStatus",
        palette="muted",
        legend=False
    )
    plt.title("Job Role Distribution", fontsize=14, weight='bold')
    plt.xlabel("Employment Status", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=30)
    st.pyplot(plt.gcf())
    plt.clf()

    # Vehicle Class Distribution
    sorted_order = df_filtered['Vehicle_Class'].value_counts().index
    plt.figure(figsize=(8,6))
    sns.set_style("white")
    ax = sns.countplot(
        data=df_filtered,
        x="Vehicle_Class",
        hue="Vehicle_Class",
        order=sorted_order,
        palette="Set3",
        legend=False
    )
    plt.title("Vehicle Class Distribution", fontsize=14, weight='bold')
    plt.xlabel("Vehicle Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Average Income by Education Level
    df_income = df_filtered.groupby('Education')['Income'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 5))
    sns.set_style("white")
    ax = sns.barplot(data=df_income, x='Education', y='Income', hue='Education', palette='pastel', dodge=False, legend=False)
    plt.title('Average Income by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Average Income (USD)')
    plt.xticks(rotation=30)
    for i, v in enumerate(df_income['Income']):
        ax.text(i, v + 200, f'{v:.0f}', color='black', ha='center')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Average CLV by Education Level
    df_clv_edu = df_filtered.groupby('Education')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 5))
    sns.set_style("white")
    ax = sns.barplot(data=df_clv_edu, x='Education', y='CLV_Corrected', hue='Education', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Education Level')
    plt.xlabel('Education')
    plt.ylabel('Average Customer Lifetime Value (Corrected)')
    for i, v in enumerate(df_clv_edu['CLV_Corrected']):
        ax.text(i, v + max(df_clv_edu['CLV_Corrected'])*0.01, f'{v:.0f}', color='black', ha='center')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Average CLV by Gender
    df_clv_gender = df_filtered.groupby('Gender')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(6, 5))
    sns.set_style("white")
    ax = sns.barplot(data=df_clv_gender, x='Gender', y='CLV_Corrected', hue='Gender', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Customer Lifetime Value (Corrected)')
    for i, v in enumerate(df_clv_gender['CLV_Corrected']):
        ax.text(i, v + max(df_clv_gender['CLV_Corrected'])*0.01, f'{v:.0f}', color='black', ha='center')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Average CLV by State
    df_clv_state = df_filtered.groupby('State')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(15, 6))
    sns.set_style("white")
    ax = sns.barplot(data=df_clv_state, x='State', y='CLV_Corrected', hue='State', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by State')
    plt.xlabel('State')
    plt.ylabel('Average Customer Lifetime Value (Corrected)')
    plt.xticks(rotation=45)
    for i, v in enumerate(df_clv_state['CLV_Corrected']):
        ax.text(i, v + max(df_clv_state['CLV_Corrected'])*0.01, f'{v:.0f}', color='black', ha='center')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Policy Count by State
    policy_count_state = df_filtered.groupby('State')['Number_of_Policies'].sum().sort_values(ascending=False)
    plt.figure(figsize=(15, 6))
    sns.set_style("white")
    ax = policy_count_state.plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title('Total Number of Policies by State')
    plt.xlabel('State')
    plt.ylabel('Total Number of Policies')
    plt.xticks(rotation=45)
    for i, v in enumerate(policy_count_state):
        ax.annotate(f'{int(v)}', (i, v), ha='center', va='bottom', fontsize=9, color='black')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Total Number of Policies by Gender
    policy_count_gender = df_filtered.groupby('Gender')['Number_of_Policies'].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    ax = policy_count_gender.plot(kind='bar', color=sns.color_palette('pastel')[0], edgecolor='black')
    plt.title('Total Number of Policies by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Total Number of Policies')
    plt.xticks(rotation=0)
    for i, v in enumerate(policy_count_gender):
        ax.annotate(f'{int(v)}', (i, v), ha='center', va='bottom', fontsize=9, color='black')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Total Number of Policies by Education
    policy_count_edu = df_filtered.groupby('Education')['Number_of_Policies'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    ax = policy_count_edu.plot(kind='bar', color=sns.color_palette('pastel')[1], edgecolor='black')
    plt.title('Total Number of Policies by Education')
    plt.xlabel('Education Level')
    plt.ylabel('Total Number of Policies')
    plt.xticks(rotation=30)
    for i, v in enumerate(policy_count_edu):
        ax.annotate(f'{int(v)}', (i, v), ha='center', va='bottom', fontsize=9, color='black')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def policy_coverage(df_filtered):
    st.header("Policy and Coverage Analysis")

    # Policy Type Distribution
    policy_type_dist = df_filtered['Policy_Type'].value_counts(normalize=True).reset_index()
    policy_type_dist.columns = ['Policy_Type', 'Percentage']
    plt.figure(figsize=(8,5))
    sns.set_style("white")
    ax = sns.barplot(data=policy_type_dist, x='Policy_Type', y='Percentage', hue='Policy_Type', palette='pastel', dodge=False, legend=False)
    plt.title('Policy Type Distribution')
    plt.xlabel('Policy Type')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Extra padding
    st.pyplot(plt.gcf())
    plt.clf()

    # Coverage Distribution Analysis
    coverage_dist = df_filtered['Coverage'].value_counts(normalize=True).reset_index()
    coverage_dist.columns = ['Coverage', 'Percentage']
    coverage_dist['Percentage'] = coverage_dist['Percentage'] * 100
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=coverage_dist, x='Coverage', y='Percentage', hue='Coverage', palette='pastel', dodge=False, legend=False)
    plt.title('Coverage Type Distribution')
    plt.xlabel('Coverage Type')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Policy Tenure Buckets Distribution
    tenure_bins = [0, 12, 36, 120]
    tenure_labels = ['<1yr', '1-3yrs', '3+yrs']
    df_filtered['Tenure_Bucket'] = pd.cut(df_filtered['Months_Since_Policy_Inception'], bins=tenure_bins, labels=tenure_labels, right=False)
    tenure_bucket_dist = df_filtered['Tenure_Bucket'].value_counts(normalize=True).sort_index().reset_index()
    tenure_bucket_dist.columns = ['Tenure_Bucket', 'Percentage']
    tenure_bucket_dist['Percentage'] = tenure_bucket_dist['Percentage'] * 100
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=tenure_bucket_dist, x='Tenure_Bucket', y='Percentage', hue='Tenure_Bucket', palette='pastel', dodge=False, legend=False)
    plt.title('Policy Tenure Buckets Distribution')
    plt.xlabel('Tenure Bucket')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=0)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Average Policy Tenure
    avg_tenure = df_filtered['Months_Since_Policy_Inception'].mean()
    st.markdown(f"**Average Policy Tenure (Months):** {avg_tenure:.2f}")

    # Policy Type vs Coverage Cross Analysis
    cross_tab = pd.crosstab(df_filtered['Policy_Type'], df_filtered['Coverage'], normalize='index') * 100
    plt.figure(figsize=(12,7))
    sns.set_style("white")
    ax = cross_tab.plot(kind='bar', stacked=True, colormap='Pastel1', edgecolor='black')
    plt.title('Policy Type vs Coverage Cross-analysis')
    plt.xlabel('Policy Type')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2, p.get_y() + height / 2),
                        ha='center', va='center', fontsize=10, color='black')
    plt.xticks(rotation=30)
    plt.legend(title='Coverage', loc='upper left', bbox_to_anchor=(1, 1))
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(plt.gcf())
    plt.clf()

    # Average Premium by Coverage Level
    avg_premium_coverage = df_filtered.groupby('Coverage')['Monthly_Premium_Auto'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=avg_premium_coverage, x='Coverage', y='Monthly_Premium_Auto', hue='Coverage', palette='pastel', dodge=False, legend=False)
    plt.title('Average Premium by Coverage Level')
    plt.xlabel('Coverage Type')
    plt.ylabel('Average Monthly Premium')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # High-value Policy Holders (Top 10% by Premium)
    premium_threshold = df_filtered['Monthly_Premium_Auto'].quantile(0.9)
    high_value = df_filtered[df_filtered['Monthly_Premium_Auto'] > premium_threshold]
    st.markdown(f"**Number of high-value policy holders (Top 10% by Premium):** {len(high_value)}")
    st.dataframe(high_value[['Customer', 'Monthly_Premium_Auto', 'Policy_Type', 'Coverage']].sort_values(by='Monthly_Premium_Auto', ascending=False))

    # Customer Segmentation by Policy & Coverage
    segmentation = df_filtered.groupby(['Policy_Type', 'Coverage']).size().reset_index(name='Customer_Count')
    plt.figure(figsize=(12,7))
    sns.set_style("white")
    ax = sns.barplot(data=segmentation, x='Policy_Type', y='Customer_Count', hue='Coverage', palette='pastel')
    plt.title('Customer Count by Policy Type & Coverage')
    plt.xlabel('Policy Type')
    plt.ylabel('Number of Customers')
    plt.legend(title='Coverage', bbox_to_anchor=(1.05, 1), loc='upper left')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(plt.gcf())
    plt.clf()

    # Premium Contribution by Policy Type
    contrib = df_filtered.groupby('Policy_Type')['Monthly_Premium_Auto'].sum().reset_index().sort_values('Monthly_Premium_Auto', ascending=False)
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=contrib, x='Policy_Type', y='Monthly_Premium_Auto', hue='Policy_Type', palette='pastel', dodge=False, legend=False)
    plt.title('Premium Contribution by Policy Type')
    plt.xlabel('Policy Type')
    plt.ylabel('Total Premium ($)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}',
                        (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Retention Potential by Tenure & Coverage
    retention = df_filtered.groupby(['Tenure_Bucket', 'Coverage'], observed=True).size().reset_index(name='Policy_Count')
    plt.figure(figsize=(12,7))
    sns.set_style("white")
    ax = sns.barplot(data=retention, x='Tenure_Bucket', y='Policy_Count', hue='Coverage', palette='pastel')
    plt.title('Policy Count by Tenure Bucket & Coverage')
    plt.xlabel('Tenure Bucket')
    plt.ylabel('Number of Policies')
    plt.legend(title='Coverage', bbox_to_anchor=(1.05, 1), loc='upper left')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=0)
    sns.despine()
    plt.tight_layout(rect=[0,0,0.85,1])
    st.pyplot(plt.gcf())
    plt.clf()

def claims_risk(df_filtered):
    st.header("Claims and Risk Analysis")

    # Total Claims Count
    total_claims = df_filtered['Number_of_Open_Complaints'].sum()
    st.markdown(f"**Total Claims Count:** {total_claims}")

    # Claim Rate (%)
    claim_rate = (df_filtered[df_filtered['Number_of_Open_Complaints'] > 0].shape[0] / df_filtered.shape[0]) * 100
    st.markdown(f"**Claim Rate (%):** {claim_rate:.2f}")

    # Average Claims per Customer
    avg_claims = df_filtered['Number_of_Open_Complaints'].mean()
    st.markdown(f"**Average Claims per Customer:** {avg_claims:.2f}")

    # Total Claim Amount
    total_claim_amount = df_filtered['Total_Claim_Amount'].sum()
    st.markdown(f"**Total Claim Amount:** ${total_claim_amount:.2f}")

    # Average Claim Amount
    avg_claim_amount = df_filtered['Total_Claim_Amount'].mean()
    st.markdown(f"**Average Claim Amount:** ${avg_claim_amount:.2f}")

    # Claims by Policy Type
    claims_by_policy = df_filtered.groupby('Policy_Type')['Total_Claim_Amount'].sum().reset_index()
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=claims_by_policy, x='Policy_Type', y='Total_Claim_Amount', hue='Policy_Type', palette='pastel', dodge=False, legend=False)
    plt.title('Total Claims by Policy Type')
    plt.xlabel('Policy Type')
    plt.ylabel('Total Claim Amount')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', 
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Claims by Vehicle Class
    claims_by_vehicle = df_filtered.groupby('Vehicle_Class')['Total_Claim_Amount'].sum().reset_index()
    claims_by_vehicle = claims_by_vehicle.sort_values(by='Total_Claim_Amount', ascending=False)
    plt.figure(figsize=(10,6))
    sns.set_style("white")
    ax = sns.barplot(data=claims_by_vehicle, x='Vehicle_Class', y='Total_Claim_Amount', hue='Vehicle_Class', palette='pastel', dodge=False, legend=False)
    plt.title('Total Claims by Vehicle Class')
    plt.xlabel('Vehicle Class')
    plt.ylabel('Total Claim Amount')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', 
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Claim Rate by Coverage Level
    claim_rate_coverage = df_filtered.groupby('Coverage', observed=True).apply(
        lambda x: (x['Number_of_Open_Complaints'] > 0).sum() / x.shape[0] * 100,
        include_groups=False
    ).reset_index(name='Claim_Rate(%)')
    claim_rate_coverage = claim_rate_coverage.sort_values(by='Claim_Rate(%)', ascending=False)
    plt.figure(figsize=(8,6))
    sns.set_style("white")
    ax = sns.barplot(data=claim_rate_coverage, x='Coverage', y='Claim_Rate(%)', hue='Coverage', palette='pastel', dodge=False, legend=False)
    plt.title('Claim Rate by Coverage Level')
    plt.xlabel('Coverage')
    plt.ylabel('Claim Rate (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Loss Ratio (%)
    loss_ratio = (df_filtered['Total_Claim_Amount'].sum() / df_filtered['Monthly_Premium_Auto'].sum()) * 100
    st.markdown(f"**Loss Ratio (%):** {loss_ratio:.2f}")

    # Risk Segment Analysis (Sorted Descending by Count)
    df_filtered['Risk_Segment'] = np.where(df_filtered['Number_of_Open_Complaints'] > 2, 'High', 'Normal')
    risk_segment_dist = df_filtered['Risk_Segment'].value_counts(normalize=True).reset_index()
    risk_segment_dist.columns = ['Risk_Segment', 'Percentage']
    risk_segment_dist['Percentage'] = risk_segment_dist['Percentage'] * 100
    risk_segment_dist = risk_segment_dist.sort_values(by='Percentage', ascending=False)
    plt.figure(figsize=(8,6))
    sns.set_style("white")
    ax = sns.barplot(data=risk_segment_dist, x='Risk_Segment', y='Percentage', hue='Risk_Segment', palette='pastel', dodge=False, legend=False)
    plt.title('Risk Segment Distribution')
    plt.xlabel('Risk Segment')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
    plt.xticks(rotation=0)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def customer_ltv(df_filtered):
    st.header("Customer Lifetime Value Analysis")

    # Average CLV
    avg_clv = df_filtered['CLV_Corrected'].mean()
    st.markdown(f"**Average CLV:** ${avg_clv:.2f}")

    # Median CLV
    median_clv = df_filtered['CLV_Corrected'].median()
    st.markdown(f"**Median CLV:** ${median_clv:.2f}")

    # Total CLV
    total_clv = df_filtered['CLV_Corrected'].sum()
    st.markdown(f"**Total CLV:** ${total_clv:.2f}")

    # CLV by Policy Type
    clv_by_policy = df_filtered.groupby('Policy_Type')['CLV_Corrected'].mean().reset_index()
    clv_by_policy = clv_by_policy.sort_values(by='CLV_Corrected', ascending=False)
    plt.figure(figsize=(7,5))
    sns.set_style("white")
    ax = sns.barplot(data=clv_by_policy, x='Policy_Type', y='CLV_Corrected', hue='Policy_Type', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Policy Type')
    plt.xlabel('Policy Type')
    plt.ylabel('Average CLV')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0,3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # CLV by Coverage Level
    clv_by_coverage = df_filtered.groupby('Coverage')['CLV_Corrected'].mean().reset_index()
    clv_by_coverage = clv_by_coverage.sort_values(by='CLV_Corrected', ascending=False)
    plt.figure(figsize=(6,5))
    sns.set_style("white")
    ax = sns.barplot(data=clv_by_coverage, x='Coverage', y='CLV_Corrected', hue='Coverage', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Coverage Level')
    plt.xlabel('Coverage')
    plt.ylabel('Average CLV')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0,3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # CLV by Customer Segment
    df_filtered['CLV_Segment'] = pd.qcut(df_filtered['CLV_Corrected'], 3, labels=['Low','Medium','High'])
    clv_segment_dist = df_filtered['CLV_Segment'].value_counts(normalize=True).reset_index()
    clv_segment_dist.columns = ['CLV_Segment', 'Percentage']
    clv_segment_dist['Percentage'] = clv_segment_dist['Percentage'] * 100
    clv_segment_dist = clv_segment_dist.sort_values(by='Percentage', ascending=False)
    plt.figure(figsize=(6,5))
    sns.set_style("white")
    ax = sns.barplot(data=clv_segment_dist, x='CLV_Segment', y='Percentage', hue='CLV_Segment', palette='pastel', dodge=False, legend=False)
    plt.title('CLV Segment Distribution')
    plt.xlabel('CLV Segment')
    plt.ylabel('Percentage (%)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0,3), textcoords='offset points')
    plt.xticks(rotation=0)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # High CLV Customers (%)
    high_clv_pct = (df_filtered[df_filtered['CLV_Corrected'] > df_filtered['CLV_Corrected'].quantile(0.75)].shape[0] / df_filtered.shape[0]) * 100
    st.markdown(f"**High CLV Customers (%):** {high_clv_pct:.2f}")

    # CLV by Sales Channel
    clv_by_channel = df_filtered.groupby('Sales_Channel')['CLV_Corrected'].mean().reset_index()
    clv_by_channel = clv_by_channel.sort_values(by='CLV_Corrected', ascending=False)
    plt.figure(figsize=(7,5))
    sns.set_style("white")
    ax = sns.barplot(data=clv_by_channel, x='Sales_Channel', y='CLV_Corrected', hue='Sales_Channel', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Sales Channel')
    plt.xlabel('Sales Channel')
    plt.ylabel('Average CLV')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0,3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # CLV vs. Premium Correlation
    clv_premium_corr = df_filtered['CLV_Corrected'].corr(df_filtered['Monthly_Premium_Auto'])
    st.markdown(f"**Correlation (CLV vs. Premium):** {clv_premium_corr:.2f}")

    # CLV Retention Impact
    clv_retention = df_filtered.groupby('Renew_Offer_Type', observed=True)['CLV_Corrected'].mean().reset_index()
    clv_retention = clv_retention.sort_values(by='CLV_Corrected', ascending=False)
    plt.figure(figsize=(8,6))
    sns.set_style("white")
    ax = sns.barplot(data=clv_retention, x='Renew_Offer_Type', y='CLV_Corrected', hue='Renew_Offer_Type', palette='pastel', dodge=False, legend=False)
    plt.title('Average CLV by Retention Offer')
    plt.xlabel('Renew Offer Type')
    plt.ylabel('Average CLV')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0,3), textcoords='offset points')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # CLV Growth Potential (Middle 25%-75% segment)
    lower_bound = df_filtered['CLV_Corrected'].quantile(0.25)
    upper_bound = df_filtered['CLV_Corrected'].quantile(0.75)
    growth_potential_customers = df_filtered[(df_filtered['CLV_Corrected'] > lower_bound) & (df_filtered['CLV_Corrected'] < upper_bound)]
    st.markdown(f"**Number of Growth Potential Customers (25%-75% segment):** {len(growth_potential_customers)}")

    # Top 10% CLV Contribution
    top_10pct_count = int(0.1 * df_filtered.shape[0])
    top_10pct_clv = df_filtered.nlargest(top_10pct_count, 'CLV_Corrected')['CLV_Corrected'].sum()
    top_10pct_clv_share = (top_10pct_clv / df_filtered['CLV_Corrected'].sum()) * 100
    st.markdown(f"**Top 10% CLV Contribution (%):** {top_10pct_clv_share:.2f}")

##-------------MARKETING AND CAMPAIGN PERFORMANCE ANALYSIS


df_insurance['Response'] = df_insurance['Response'].astype(str)


# Response Rate by Channel
response_by_channel = df_insurance.groupby('Sales_Channel').apply(
    lambda x: (x['Response'] == 'True').mean() * 100,
    include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

response_by_channel['Sales_Channel'] = response_by_channel['Sales_Channel'].astype(str)

plt.figure(figsize=(10,6))
sns.set_style("white")
ax = sns.barplot(
    data=response_by_channel,
    x='Sales_Channel',
    y='Response_Rate(%)',
    hue='Sales_Channel',
    palette='pastel',
    dodge=False,
    legend=False
)
plt.title('Response Rate by Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Response Rate (%)')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=12, color='black',
                xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


# Response Rate by Policy Type
response_by_policy = df_insurance.groupby('Policy_Type').apply(
    lambda x: (x['Response'] == 'True').mean() * 100,
    include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

plt.figure(figsize=(6,5))
sns.set_style("white")
ax = sns.barplot(data=response_by_policy, x='Policy_Type', y='Response_Rate(%)', color='lightgreen')
plt.title('Response Rate by Policy Type')
plt.xlabel('Policy Type')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


# Response Rate by Coverage
response_by_coverage = df_insurance.groupby('Coverage').apply(
    lambda x: (x['Response'] == 'True').mean() * 100,
    include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

plt.figure(figsize=(6,5))
sns.set_style("white")
ax = sns.barplot(data=response_by_coverage, x='Coverage', y='Response_Rate(%)', color='lightcoral')
plt.title('Response Rate by Coverage')
plt.xlabel('Coverage')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


# Response Rate by CLV Segment
df_insurance['CLV_Segment'] = pd.qcut(df_insurance['CLV_Corrected'], 3, labels=['Low', 'Medium', 'High'])
response_by_clv = df_insurance.groupby('CLV_Segment', observed=True).apply(
    lambda x: (x['Response'] == 'True').mean() * 100,
    include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

plt.figure(figsize=(6,5))
sns.set_style("white")
ax = sns.barplot(data=response_by_clv, x='CLV_Segment', y='Response_Rate(%)', color='lightskyblue')
plt.title('Response Rate by CLV Segment')
plt.xlabel('CLV Segment')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.show()


# Offer Type Performance
offers = ['Offer1', 'Offer2', 'Offer3', 'Offer4']
response_by_offer = df_insurance.groupby('Renew_Offer_Type').apply(
    lambda x: (x['Response'] == 'True').mean() * 100,
    include_groups=False
).reset_index(name='Response_Rate(%)')
response_by_offer = response_by_offer.set_index('Renew_Offer_Type').reindex(offers, fill_value=0).reset_index()

plt.figure(figsize=(8,6))
sns.set_style("white")
ax = sns.barplot(data=response_by_offer, x='Renew_Offer_Type', y='Response_Rate(%)', color='lightpink')
plt.title('Response Rate by Offer Type')
plt.xlabel('Renew Offer Type')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=12, color='black',
                xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


# Income Bracket Response Rate
df_insurance['Income_Bracket'] = pd.cut(df_insurance['Income'], bins=[-1, 25000, 50000, 75000, 100000, float('inf')],
                                        labels=['<25K','25-50K','50-75K','75-100K','100K+'])
income_response = df_insurance.groupby('Income_Bracket', observed=True).apply(
    lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)
plt.figure(figsize=(10,6))
sns.set_style("white")
ax = sns.barplot(data=income_response, x='Income_Bracket', y='Response_Rate(%)', color='lightblue')
plt.title('Response Rate by Income Bracket')
plt.xlabel('Income Bracket')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()


# Top Converting Segment (Policy Type x Coverage)
top_segment = df_insurance.groupby(['Policy_Type', 'Coverage'], observed=True).apply(
    lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)

plt.figure(figsize=(8,7))
sns.set_style("whitegrid")

# Sort DataFrame descending by Response Rate for clear legend ordering
top_segment_sorted = top_segment.sort_values(by='Response_Rate(%)', ascending=False)

# Plot grouped barplot by Coverage with hue=Policy_Type
ax = sns.barplot(data=top_segment_sorted, x='Coverage', y='Response_Rate(%)', hue='Policy_Type', palette='pastel')

plt.title('Top Converting Segments: Response Rate by Policy Type and Coverage')
plt.xlabel('Coverage Level')
plt.ylabel('Response Rate (%)')

for p in ax.patches:
    height = p.get_height()
    if height > 0.05:  # Only annotate bars with height greater than 0.05%
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=11, color='black',
                    xytext=(0,3), textcoords='offset points')

plt.legend(title='Policy Type')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Retention vs. New Acquisition Response
df_insurance['Is_New'] = df_insurance['Months_Since_Policy_Inception'] < 12
retention_response = df_insurance.groupby('Is_New', observed=True).apply(
    lambda x: (x['Response'] == 'True').mean() * 100, include_groups=False
).reset_index(name='Response_Rate(%)').sort_values(by='Response_Rate(%)', ascending=False)
retention_response['Customer_Type'] = retention_response['Is_New'].map({True:'New', False:'Retention'})
plt.figure(figsize=(5,6))
sns.set_style("white")
ax = sns.barplot(data=retention_response, x='Customer_Type', y='Response_Rate(%)', color='lightcoral')
plt.title('Response Rate: Retention vs. New Acquisition')
plt.xlabel('Customer Type')
plt.ylabel('Response Rate (%)')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width()/2, height), ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()







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