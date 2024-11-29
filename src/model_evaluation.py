import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import common

def evaluate_model(model, x_test, y_test):
    print(f"Evaluating the model")    
    y_pred_test = model.predict(x_test)

    print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
    print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))

