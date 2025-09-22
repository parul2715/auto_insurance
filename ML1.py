import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

sns.set(style="white")

def load_data_from_sql():
    server_name = "LAPTOP-1FURGFHN\\SQLEXPRESS"
    database_name = "auto_insurance"
    driver = "ODBC+Driver+17+for+SQL+Server"

    connection_string = (
        f"mssql+pyodbc://{server_name}/{database_name}"
        f"?driver={driver}&trusted_connection=yes"
    )
    engine = create_engine(connection_string)
    query = "SELECT * FROM insurance"  # Adjust if needed
    df = pd.read_sql(query, engine)
    return df

def data_prep_for_modeling(df):
    # Encode target variable to binary
    df['Response_Binary'] = df['Response'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

    categorical_cols = [
        'State', 'Coverage', 'Education', 'EmploymentStatus', 'Gender',
        'Location_Code', 'Marital_Status', 'Policy_Type', 'Policy',
        'Renew_Offer_Type', 'Sales_Channel', 'Vehicle_Class', 'Vehicle_Size'
    ]

    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if 'Effective_To_Date' in df.columns:
        df['Effective_To_Date'] = pd.to_datetime(df['Effective_To_Date'], errors='coerce')
        df['Effective_To_Date'] = df['Effective_To_Date'].astype('int64') // 10**9  # Unix timestamp

    numeric_cols = [
        'Income',
        'Monthly_Premium_Auto',
        'Months_Since_Last_Claim',
        'Months_Since_Policy_Inception',
        'Number_of_Open_Complaints',
        'Number_of_Policies',
        'Total_Claim_Amount',
        'CLV_Corrected'
    ]
    numeric_cols_present = [col for col in numeric_cols if col in df.columns]
    scaler = StandardScaler()
    df[numeric_cols_present] = scaler.fit_transform(df[numeric_cols_present])

    # Save encoders and scaler for reuse
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    exclude_cols = ['Customer', 'Response', 'Response_Binary','Customer_Lifetime_Value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['Response_Binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, encoders, scaler

def main():
    df = load_data_from_sql()
    X_train, X_test, y_train, y_test, encoders, scaler = data_prep_for_modeling(df)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    print("Training accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    print("Starting Grid Search for hyperparameter tuning... This might take some time.")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best hyperparameters found:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    importances = best_model.feature_importances_
    feature_names = X_train.columns

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    feat_imp.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

    # Save the best model
    joblib.dump(best_model, 'random_forest_best_model.pkl')
    print("Best model saved as 'random_forest_best_model.pkl'")

if __name__ == "__main__":
    main()