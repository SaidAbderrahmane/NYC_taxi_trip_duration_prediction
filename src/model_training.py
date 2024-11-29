import os
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

    model = pipeline.fit(X_train[train_features], y_train)
    y_pred_train = model.predict(X_train[train_features])

    print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))

    return model

def save_model(model, path):
    # Save the model using pickle
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {path}")
    

def main():
    from data_preprocessing import load_data, preprocess_data, split_data
    df = load_data(common.DB_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model,common.MODEL_PATH)
    evaluate_model(model, X_train, y_train)

if __name__ == "__main__":
    main()
