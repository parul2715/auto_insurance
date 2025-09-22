import pandas as pd
from sqlalchemy import create_engine

# Your Render PostgreSQL database URL
db_url = "postgresql://auto_insurance_user:e1jQYbP4i70gEXeIr8rs6RV22Bjh8vDj@dpg-d38srq7fte5s73ccgra0-a.oregon-postgres.render.com/auto_insurance"
# Create SQLAlchemy engine
engine = create_engine(db_url)

# Path to your insurance CSV file
insurance_csv_path = r"C:\Users\parul\OneDrive\Desktop\auto insurance project\insurance.csv"

# Load CSV into DataFrame
df_insurance = pd.read_csv(insurance_csv_path)

# Upload DataFrame to PostgreSQL - replace table if it exists
df_insurance.to_sql('insurance', engine, if_exists='replace', index=False)

print("✅ Uploaded 'insurance' table to PostgreSQL successfully!")