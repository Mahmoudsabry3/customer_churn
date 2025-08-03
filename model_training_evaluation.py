import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

def train_and_evaluate_model(file_path):
    df = pd.read_csv(file_path)

    # Define features (X) and target (y)
    X = df.drop(columns=["userId", "churn"])
    y = df["churn"]

    # Identify numerical features
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

    # Create a preprocessor to scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features)
        ])

    # Define the base model pipeline
    base_model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False, # Suppress warning
            random_state=42
        ))
    ])

    # --- K-Fold Cross-Validation for Robust Evaluation ---
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_test = []
    all_y_pred = []
    all_y_proba = []

    print("\n--- Starting K-Fold Cross-Validation ---")
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/{kf.n_splits}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Clone the pipeline for each fold to ensure fresh training
        fold_model_pipeline = base_model_pipeline
        fold_model_pipeline.fit(X_train, y_train)

        y_pred_fold = fold_model_pipeline.predict(X_test)
        y_proba_fold = fold_model_pipeline.predict_proba(X_test)[:, 1]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred_fold)
        all_y_proba.extend(y_proba_fold)

    # Convert lists to numpy arrays for final evaluation
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # --- Aggregate Evaluation Metrics ---
    accuracy = accuracy_score(all_y_test, all_y_pred)
    precision = precision_score(all_y_test, all_y_pred)
    recall = recall_score(all_y_test, all_y_pred)
    f1 = f1_score(all_y_test, all_y_pred)
    roc_auc = roc_auc_score(all_y_test, all_y_proba)

    prec, rec, _ = precision_recall_curve(all_y_test, all_y_proba)
    pr_auc = auc(rec, prec)

    print("\n--- Aggregated Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # --- Error Analysis ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_y_test, all_y_pred)
    print(cm)
    print("  True Negative (TN):", cm[0, 0])
    print("  False Positive (FP):", cm[0, 1])
    print("  False Negative (FN):", cm[1, 0])
    print("  True Positive (TP):", cm[1, 1])

    print("\n--- Classification Report ---")
    print(classification_report(all_y_test, all_y_pred, target_names=["No Churn", "Churn"])) # Added target names

    # --- Business Implications of Errors ---
    # False Positives (FP): Predicted Churn, Actual No Churn
    #   - Business Impact: Unnecessary intervention costs (e.g., retention offers, customer support outreach) for users who would not have churned anyway.
    #   - Strategy: If FP is high, focus on refining targeting or adjusting prediction threshold.
    # False Negatives (FN): Predicted No Churn, Actual Churn
    #   - Business Impact: Lost revenue from churned users who were not identified and retained. Missed opportunity for intervention.
    #   - Strategy: If FN is high, focus on improving model sensitivity to churn signals, potentially by adjusting threshold or feature engineering.
    print("\n--- Business Implications of Model Errors ---")
    print("False Positives (Predicted Churn, Actual No Churn):", cm[0, 1])
    print("  - Business Impact: Potential wasted resources on retention efforts for users who would not have churned.")
    print("False Negatives (Predicted No Churn, Actual Churn):", cm[1, 0])
    print("  - Business Impact: Lost revenue and missed opportunities to retain valuable customers.")

    # --- Save the trained model (using the last fold's model for simplicity for deployment) ---
    # In a real-world scenario, you might retrain on the full dataset or use an ensemble of models.
    final_model_pipeline = base_model_pipeline # This will be the model trained on the last fold
    final_model_pipeline.fit(X, y) # Retrain on full dataset for deployment
    joblib.dump(final_model_pipeline, "churn_prediction_model.pkl")
    print("\nTrained model saved to churn_prediction_model.pkl")

    # Save evaluation metrics and error analysis to a file
    with open("model_evaluation_metrics.txt", "w") as f:
        f.write("--- Aggregated Model Evaluation Results ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n")
        f.write("\n--- Confusion Matrix ---\n")
        f.write(str(cm) + "\n")
        f.write(f"  True Negative (TN): {cm[0, 0]}\n")
        f.write(f"  False Positive (FP): {cm[0, 1]}\n")
        f.write(f"  False Negative (FN): {cm[1, 0]}\n")
        f.write(f"  True Positive (TP): {cm[1, 1]}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(classification_report(all_y_test, all_y_pred, target_names=["No Churn", "Churn"])) # Added target names
        f.write("\n--- Business Implications of Model Errors ---\n")
        f.write(f"False Positives (Predicted Churn, Actual No Churn): {cm[0, 1]}\n")
        f.write("  - Business Impact: Potential wasted resources on retention efforts for users who would not have churned.\n")
        f.write(f"False Negatives (Predicted No Churn, Actual Churn): {cm[1, 0]}\n")
        f.write("  - Business Impact: Lost revenue and missed opportunities to retain valuable customers.\n")

if __name__ == "__main__":
    featured_data_path = "featured_data.csv"
    train_and_evaluate_model(featured_data_path)


