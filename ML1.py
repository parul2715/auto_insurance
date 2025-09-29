import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
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
    query = "SELECT * FROM insurance"
    df = pd.read_sql(query, engine)
    return df

def data_prep_for_modeling(df):
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
        df['Effective_To_Date'] = df['Effective_To_Date'].astype('int64') // 10**9

    numeric_cols = [
        'Income', 'Monthly_Premium_Auto', 'Months_Since_Last_Claim',
        'Months_Since_Policy_Inception', 'Number_of_Open_Complaints',
        'Number_of_Policies', 'Total_Claim_Amount', 'CLV_Corrected'
    ]
    numeric_cols_present = [col for col in numeric_cols if col in df.columns]
    scaler = StandardScaler()
    df[numeric_cols_present] = scaler.fit_transform(df[numeric_cols_present])

    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    exclude_cols = ['Customer', 'Response', 'Response_Binary', 'Customer_Lifetime_Value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['Response_Binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, encoders, scaler

def print_classification_metrics(y_true, y_pred, y_proba, model_name):
    print(f"\n🔍 Metrics for {model_name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_proba))

def main():
    df = load_data_from_sql()
    X_train, X_test, y_train, y_test, encoders, scaler = data_prep_for_modeling(df)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    baseline = LogisticRegression(max_iter=500, random_state=42)
    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    print_classification_metrics(y_test, baseline_pred, baseline_proba, "Logistic Regression")

    # Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("Running GridSearchCV for Random Forest...")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print_classification_metrics(y_test, y_pred, y_proba, "Random Forest")

    # Feature Importance
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

    # Model Comparison Table
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, baseline_pred),
            accuracy_score(y_test, y_pred)
        ],
        "Precision": [
            precision_score(y_test, baseline_pred),
            precision_score(y_test, y_pred)
        ],
        "Recall": [
            recall_score(y_test, baseline_pred),
            recall_score(y_test, y_pred)
        ],
        "F1-Score": [
            f1_score(y_test, baseline_pred),
            f1_score(y_test, y_pred)
        ],
        "ROC AUC": [
            roc_auc_score(y_test, baseline_proba),
            roc_auc_score(y_test, y_proba)
        ]
    })

    print("\n📊 Model Performance Comparison:\n", comparison)

    # Visualize Comparison
    comparison_melted = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=comparison_melted, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Model Performance Comparison", fontsize=14, weight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=10, color='white', weight='bold')

    plt.show()

    # Save models
    joblib.dump(baseline, 'baseline_logistic_regression.pkl')
    joblib.dump(best_model, 'random_forest_best_model.pkl')
    joblib.dump(comparison, "model_comparison.pkl")
    print("✅ Models saved successfully.")

if __name__ == "__main__":
    main()