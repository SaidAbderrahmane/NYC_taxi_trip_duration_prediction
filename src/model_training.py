import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn

import common
from model_evaluation import evaluate_model

def train_model(X_train, y_train):
    num_features = ['hour']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    with mlflow.start_run(run_name="Ridge", description="Ridge") as run:
        model = pipeline.fit(X_train[train_features], y_train)
        y_pred_train = model.predict(X_train[train_features])
        
        # Infer an MLflow model signature from the training data (input),
        # model predictions (output) and parameters (for inference).
        signature = mlflow.models.infer_signature(X_train, y_train)
  
               
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        print("Train RMSE = %.4f" % train_rmse)

        mlflow.log_metric("train_rmse", train_rmse)
        # Log model
        model=  mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature)

    return model,run

def save_model(model, path):
    # Save the model using pickle
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {path}")
    
def main():

    DIR_MLRUNS = os.path.join(common.ROOT_DIR, "mlruns")
    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)
    mlflow.set_experiment("NYC_Taxi_Trip_Prediction")

    from data_preprocessing import load_data, preprocess_data, split_data
    df = load_data(common.DB_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model, run = train_model(X_train, y_train)
    save_model(model,common.MODEL_PATH)
    evaluate_model(model, X_test, y_test)
    
    model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    print(model_uri)
    mv = mlflow.register_model(model_uri, "NYC_Taxi_Trip_Prediction")
    print(f"Model registered with name: {mv.name}")
    print(f"Model version: {mv.version}")
    print(f"Source: {mv.source}")

    mlflow.end_run()


if __name__ == "__main__":
    main()

