
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Step 2: Define your SQL Server connection details ---
server_name = "LAPTOP-1FURGFHN\\SQLEXPRESS"
database_name = "auto_insurance"
driver = "ODBC+Driver+17+for+SQL+Server"

# --- Step 3: Create the connection string ---
connection_string = (
    f"mssql+pyodbc://{server_name}/{database_name}"
    f"?driver={driver}&trusted_connection=yes"
)

# --- Step 4: Create SQLAlchemy engine ---
engine = create_engine(connection_string)

# --- Step 5: Load the 'insurance' table into a pandas DataFrame ---
query = "SELECT * FROM insurance"
df_insurance = pd.read_sql(query, engine)

# --- Step 6: Preview the data ---
print("✅ Data loaded successfully!")
print(df_insurance.head())

##------------DATA PREP-------------------

def python_data_prep(df):
    # Encode Response to binary
    df['Response_Binary'] = df['Response'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

    # Encode categorical features needed for both analysis and modeling
    categorical_cols = ['Education', 'Policy_Type', 'Sales_Channel', 'Coverage']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Optional: scale numeric columns (e.g., Monthly_Premium_Auto, Customer_Age)
    numeric_cols = ['Monthly_Premium_Auto', 'Customer_Age']
    scaler = StandardScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    
    return df


# ##--------OVERALL BUSINESS SNAPSHOT------------------


##TOTAL CUSTOMERS
total_customers = df_insurance['Customer'].nunique()
print( total_customers)


##TOTAL PREMIUM REVENUE
total_premium_revenue = df_insurance['Monthly_Premium_Auto'].sum()
print( total_premium_revenue)


##AVG PREMIUM PER CUSTOMER
avg_premium_per_customer = df_insurance['Monthly_Premium_Auto'].mean()
print(avg_premium_per_customer)


##TOTAL CLAIMS AND CLAIM RATE
total_claims = df_insurance['Total_Claim_Amount'].sum()
claim_rate = (df_insurance['Total_Claim_Amount'] > 0).mean()
print("Total Claims Amount:", total_claims)
print("Claim Rate:", claim_rate)


##CAMPAIGN RESPONSE RATE
campaign_response_rate = (df_insurance['Response'] == 1).mean()
print(campaign_response_rate)


##AVG CLV
avg_clv = df_insurance['CLV_Corrected'].mean()
print( avg_clv)


# ##----------------CUSTOMER DEMOGRAPHICS ANALYSIS-------------------


##Customer Distribution by State

state_counts = df_insurance['State'].value_counts()

plt.figure(figsize=(10,6))
ax = sns.barplot(
    x=state_counts.index,
    y=state_counts.values,
    hue=state_counts.index,   # assign x as hue
    palette="viridis",
    legend=False              # disable redundant legend
)

# Labels & title
plt.title("Customer Distribution by State", fontsize=14, weight='bold')
plt.xlabel("State", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)

# Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.show()


##Gender Distribution

plt.figure(figsize=(6,6))
sns.set_style("white")
ax = sns.countplot(
    data=df_insurance,
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

plt.show()


##Marital Status Distribution

plt.figure(figsize=(6,6))
sns.set_style("white")
ax = sns.countplot(
    data=df_insurance,
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

plt.show()


##Education Level Breakdown

plt.figure(figsize=(10,6))
sns.set_style("white")
ax = sns.countplot(
    data=df_insurance,
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
plt.show()


##Income Distribution

plt.figure(figsize=(8,6))
ax = sns.histplot(
    data=df_insurance,
    x="Income",
    bins=30,
    kde=True,
    color="teal"
)

plt.title("Income Distribution", fontsize=14, weight='bold')
plt.xlabel("Annual Income", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Add data labels on top of each bin
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # only label non-empty bins
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8)

plt.show()


##Job Role Distribution

plt.figure(figsize=(10,6))
ax = sns.countplot(
    data=df_insurance,
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
plt.show()


##vehicle class distribution
sorted_order = df_insurance['Vehicle_Class'].value_counts().index


plt.figure(figsize=(8,6))
sns.set_style("white")
ax = sns.countplot(
    data=df_insurance,
    x="Vehicle_Class",
    hue="Vehicle_Class",         
    order=sorted_order,          
    palette="Set3",
    legend=False                 
)

# Step 3: Labels and title
plt.title("Vehicle Class Distribution", fontsize=14, weight='bold')
plt.xlabel("Vehicle Class", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Step 4: Annotate bar counts
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


##AVG INCOME BY EDUCATION LEVEL

df_income = df_insurance.groupby('Education')['Income'].mean().sort_values(ascending=False).reset_index()

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
plt.show()


##Average CLV by Education Level
df_clv_edu = df_insurance.groupby('Education')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()

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
plt.show()


##avg clv by gender
df_clv_gender = df_insurance.groupby('Gender')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()

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
plt.show()


##avg clv by state
df_clv_state = df_insurance.groupby('State')['CLV_Corrected'].mean().sort_values(ascending=False).reset_index()

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
plt.show()


##POLICY COUNT BY STATE
policy_count_state = df_insurance.groupby('State')['Number_of_Policies'].sum().sort_values(ascending=False)

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
plt.show()


##Total Number of Policies by Gender
policy_count_gender = df_insurance.groupby('Gender')['Number_of_Policies'].sum().sort_values(ascending=False)

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
plt.show()


##Total Number of Policies by Education 
policy_count_edu = df_insurance.groupby('Education')['Number_of_Policies'].sum().sort_values(ascending=False)

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
plt.show()


# ##----------------POLICY AND COVERAGE ANALYSIS---------------


##Policy Type Distribution

policy_type_dist = df_insurance['Policy_Type'].value_counts(normalize=True).reset_index()
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
plt.subplots_adjust(bottom=0.2)  # Adds extra bottom padding to prevent label cutoff
plt.show()


##COVERAGE DISTIBUTION ANALYSIS
coverage_dist = df_insurance['Coverage'].value_counts(normalize=True).reset_index()
coverage_dist.columns = ['Coverage', 'Percentage']
coverage_dist['Percentage'] = coverage_dist['Percentage'] * 100  # to percentage scale

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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##POLICY TENURE BUCKETS DISTRIBUTION
tenure_bins = [0, 12, 36, 120]
tenure_labels = ['<1yr', '1-3yrs', '3+yrs']

df_insurance['Tenure_Bucket'] = pd.cut(df_insurance['Months_Since_Policy_Inception'], bins=tenure_bins, labels=tenure_labels, right=False)
tenure_bucket_dist = df_insurance['Tenure_Bucket'].value_counts(normalize=True).sort_index().reset_index()
tenure_bucket_dist.columns = ['Tenure_Bucket', 'Percentage']
tenure_bucket_dist['Percentage'] = tenure_bucket_dist['Percentage'] * 100  # to percentage

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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.show()


##AVG PLOICY TENURE
avg_tenure = df_insurance['Months_Since_Policy_Inception'].mean()
print(f'{avg_tenure:.2f}')


##POLICY TYPE VS COVERAGE CROSS ANALYSIS
cross_tab = pd.crosstab(df_insurance['Policy_Type'], df_insurance['Coverage'], normalize='index') * 100

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
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend
plt.show()


avg_premium_coverage = df_insurance.groupby('Coverage')['Monthly_Premium_Auto'].mean().sort_values(ascending=False).reset_index()

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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##High-value Policy Holders (Top 10% by Premium)
premium_threshold = df_insurance['Monthly_Premium_Auto'].quantile(0.9)
high_value = df_insurance[df_insurance['Monthly_Premium_Auto'] > premium_threshold]

print(f"Number of high-value policy holders: {len(high_value)}")
print(high_value[['Customer', 'Monthly_Premium_Auto', 'Policy_Type', 'Coverage']].sort_values(by='Monthly_Premium_Auto', ascending=False))



##Customer Segmentation by Policy & Coverage
segmentation = df_insurance.groupby(['Policy_Type', 'Coverage']).size().reset_index(name='Customer_Count')

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
                    ha='center', va='bottom', fontsize=11, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


##Premium Contribution by Policy Type
contrib = df_insurance.groupby('Policy_Type')['Monthly_Premium_Auto'].sum().reset_index().sort_values('Monthly_Premium_Auto', ascending=False)

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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##Retention Potential by Tenure & Coverage
retention = df_insurance.groupby(['Tenure_Bucket', 'Coverage'], observed=True).size().reset_index(name='Policy_Count')

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
                    ha='center', va='bottom', fontsize=11, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


# ##-------------CLAIMS AND RISK ANALYSIS--------------------


##Total Claims Count
total_claims = df_insurance['Number_of_Open_Complaints'].sum()
print(f"Total Claims Count: {total_claims}")


##Claim Rate (%)
claim_rate = (df_insurance[df_insurance['Number_of_Open_Complaints'] > 0].shape[0] / df_insurance.shape[0]) * 100
print(f"Claim Rate (%): {claim_rate:.2f}")


##Average Claims per Customer
avg_claims = df_insurance['Number_of_Open_Complaints'].mean()
print(f"Average Claims per Customer: {avg_claims:.2f}")


##Total Claim Amount
total_claim_amount = df_insurance['Total_Claim_Amount'].sum()
print(f"Total Claim Amount: {total_claim_amount:.2f}")

##Average Claim Amount
avg_claim_amount = df_insurance['Total_Claim_Amount'].mean()
print(f"Average Claim Amount: {avg_claim_amount:.2f}")



##CLAIMS BY POLICY TYPE
claims_by_policy = df_insurance.groupby('Policy_Type')['Total_Claim_Amount'].sum().reset_index()

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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')
plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##Claims by Vehicle Class
claims_by_vehicle = df_insurance.groupby('Vehicle_Class')['Total_Claim_Amount'].sum().reset_index()
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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##Claim Rate by Coverage Level
claim_rate_coverage = df_insurance.groupby('Coverage', observed=True).apply(
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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=30)
sns.despine()
plt.tight_layout()
plt.show()


##Loss Ratio (%)
loss_ratio = (df_insurance['Total_Claim_Amount'].sum() / df_insurance['Monthly_Premium_Auto'].sum()) * 100
print(f"Loss Ratio (%): {loss_ratio:.2f}")


##Risk Segment Analysis (Sorted Descending by Count)

df_insurance['Risk_Segment'] = np.where(df_insurance['Number_of_Open_Complaints'] > 2, 'High', 'Normal')
risk_segment_dist = df_insurance['Risk_Segment'].value_counts(normalize=True).reset_index()
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
                    ha='center', va='bottom', fontsize=12, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.show()

##-----------------clv analysis

# Average CLV
avg_clv = df_insurance['CLV_Corrected'].mean()
print(f"Average CLV: {avg_clv:.2f}")


# Median CLV
median_clv = df_insurance['CLV_Corrected'].median()
print(f"Median CLV: {median_clv:.2f}")




# Total CLV
total_clv = df_insurance['CLV_Corrected'].sum()
print(f"Total CLV: {total_clv:.2f}")


# CLV by Policy Type
clv_by_policy = df_insurance.groupby('Policy_Type')['CLV_Corrected'].mean().reset_index()
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
plt.show()


# CLV by Coverage Level
clv_by_coverage = df_insurance.groupby('Coverage')['CLV_Corrected'].mean().reset_index()
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
plt.show()


# CLV by Customer Segment
df_insurance['CLV_Segment'] = pd.qcut(df_insurance['CLV_Corrected'], 3, labels=['Low','Medium','High'])
clv_segment_dist = df_insurance['CLV_Segment'].value_counts(normalize=True).reset_index()
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
plt.show()


# High CLV Customers (%)
high_clv_pct = (df_insurance[df_insurance['CLV_Corrected'] > df_insurance['CLV_Corrected'].quantile(0.75)].shape[0] / df_insurance.shape[0]) * 100
print(f"High CLV Customers (%): {high_clv_pct:.2f}")


# CLV by Channel
clv_by_channel = df_insurance.groupby('Sales_Channel')['CLV_Corrected'].mean().reset_index()
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
plt.show()


## CLV vs. Premium Correlation
clv_premium_corr = df_insurance['CLV_Corrected'].corr(df_insurance['Monthly_Premium_Auto'])
print(f"Correlation (CLV vs. Premium): {clv_premium_corr:.2f}")

# 4. CLV Retention Impact
clv_retention = df_insurance.groupby('Renew_Offer_Type', observed=True)['CLV_Corrected'].mean().reset_index()
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
plt.show()


##CLV Growth Potential (Middle 25%-75% segment)
lower_bound = df_insurance['CLV_Corrected'].quantile(0.25)
upper_bound = df_insurance['CLV_Corrected'].quantile(0.75)
growth_potential_customers = df_insurance[(df_insurance['CLV_Corrected'] > lower_bound) & (df_insurance['CLV_Corrected'] < upper_bound)]
print("Number of growth potential customers:", len(growth_potential_customers))

# %%
##Top 10% CLV Contribution
top_10pct_count = int(0.1 * df_insurance.shape[0])
top_10pct_clv = df_insurance.nlargest(top_10pct_count, 'CLV_Corrected')['CLV_Corrected'].sum()
top_10pct_clv_share = (top_10pct_clv / df_insurance['CLV_Corrected'].sum()) * 100
print(f"Top 10% CLV Contribution (%): {top_10pct_clv_share:.2f}")


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





