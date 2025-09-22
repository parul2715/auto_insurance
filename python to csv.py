import pandas as pd
from sqlalchemy import create_engine

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

query = "SELECT * FROM insurance"
df = pd.read_sql(query, engine)

output_path = r"C:\Users\parul\OneDrive\Desktop\auto insurance project\insurance.csv"
df.to_csv(output_path, index=False)
print(f"CSV saved to {output_path}")