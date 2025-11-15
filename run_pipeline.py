
import mlflow
import mlflow.sklearn
from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi
import os

def run_pipeline():
    # Load dataset
    dataset = load_dataset("Vishnu-J-S/engine-data")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    X_train = train_df.drop('Engine Condition', axis=1)
    y_train = train_df['Engine Condition']
    X_test = test_df.drop('Engine Condition', axis=1)
    y_test = test_df['Engine Condition']

    # Model hyperparameters grid
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("engine-failure-prediction")

    with mlflow.start_run():
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('precision', precision_score(y_test, y_pred))
        mlflow.log_metric('recall', recall_score(y_test, y_pred))
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred))

        input_example = X_train.iloc[:1]
        mlflow.sklearn.log_model(best_rf, name="random_forest_model", input_example=input_example)

        print("Best RF Params:", grid_search.best_params_)
        print("F1 Score:", f1_score(y_test, y_pred))

    joblib.dump(best_rf, "best_rf_model.joblib")

    HF_TOKEN = os.getenv("HF_TOKEN)
    api = HfApi(token=HFTOKEN)
    repo_id = "Vishnu-J-S/engine-failure-rf-model"

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj="best_rf_model.joblib",
        path_in_repo="best_rf_model.joblib",
        repo_id=repo_id,
        repo_type="model"
    )

if __name__ == "__main__":
    run_pipeline()
  