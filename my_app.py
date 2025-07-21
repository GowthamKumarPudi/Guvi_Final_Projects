
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Streamlit config
st.set_page_config(page_title="NBFC Loan Default Predictor", layout="wide")
st.title("NBFC Vehicle Loan Default Prediction App")

# File upload section
train_file = st.file_uploader("📤Upload Train_Dataset.csv", type="csv")
test_file = st.file_uploader("📤Upload Test_Dataset.csv", type="csv")

# Run pipeline if files uploaded
if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    st.success("✅Files uploaded successfully!")

    if st.button("🚀Train Model and Predict"):

        # Drop high-missing columns
        drop_cols = ['Own_House_Age', 'Score_Source_1', 'Social_Circle_Default']
        for col in drop_cols:
            if col in train_df.columns: train_df.drop(columns=col, inplace=True)
            if col in test_df.columns: test_df.drop(columns=col, inplace=True)

        # Convert numeric-like columns
        numeric_cols = ['Client_Income', 'Credit_Amount', 'Loan_Annuity',
                        'Population_Region_Relative', 'Age_Days', 'Employed_Days',
                        'Registration_Days', 'ID_Days', 'Score_Source_3', 'Score_Source_2']
        for col in numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col].astype(str).str.replace(",", ""), errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col].astype(str).str.replace(",", ""), errors='coerce')

        # Handle missing values
        numerical = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Default' in numerical: numerical.remove('Default')
        categorical = train_df.select_dtypes(include='object').columns.tolist()

        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        train_df[numerical] = num_imputer.fit_transform(train_df[numerical])
        test_df[numerical] = num_imputer.transform(test_df[numerical])
        train_df[categorical] = cat_imputer.fit_transform(train_df[categorical])
        test_df[categorical] = cat_imputer.transform(test_df[categorical])

        # Encode categorical features
        encoders = {}
        for col in categorical:
            le = LabelEncoder()
            all_vals = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
            le.fit(all_vals)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le

        # Split train data
        X = train_df.drop("Default", axis=1)
        y = train_df["Default"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(test_df)

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_val, y_proba)

        # 📊Confusion Matrix
        st.subheader("📊Model Evaluation on Validation Set")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # 📈Classification Report
        st.markdown("### Classification Report Summary")
        metrics_df = pd.DataFrame(report).transpose()
        st.dataframe(metrics_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        }).highlight_max(axis=0, color='lightgreen'))

        # ROC AUC
        st.markdown(f"### 🧮 ROC AUC Score: **{roc_auc:.4f}**")

        # 📘Interpretation
        st.markdown("""
#### 📘Quick Interpretation:
- **Precision**: Out of predicted defaults, how many were correct? (Low means many false positives)
- **Recall**: Out of actual defaults, how many did we catch? (Important in risk modeling)
- **F1-score**: Balances precision and recall.
- **ROC AUC**: Area under the curve — 0.73 means decent discrimination between defaulters vs. non-defaulters.
""")

        # Prediction
        test_preds = model.predict(X_test_scaled)
        possible_ids = [col for col in test_df.columns if 'id' in col.lower()]
        id_col = possible_ids[0] if possible_ids else None

        submission = pd.DataFrame({
            "UniqueID": test_df[id_col] if id_col else test_df.index,
            "Default": test_preds
        })

        csv_buffer = io.StringIO()
        submission.to_csv(csv_buffer, index=False)
        st.download_button("📥Download Prediction CSV", data=csv_buffer.getvalue(), file_name="vehicle_loan_default_predictions.csv", mime='text/csv')
