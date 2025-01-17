import sqlite3
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import common

def evaluate_model(model, x_test, y_test):
    
    print(f"Evaluating the model")    
    # y_pred_test = model.predict(x_test)
    # print("evaluation result:")
    # print(x_test.head(5))
    # print(y_test .head(5))
    # print(y_pred_test)
    # print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
    # print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))
    df = pd.concat([x_test,y_test], axis=1)
    print(df.head(5))
    results = mlflow.evaluate(
        model.model_uri,
        data=df,
        targets="log_trip_duration",
        model_type="regressor",
        evaluators=["default"]
    )
    return results
