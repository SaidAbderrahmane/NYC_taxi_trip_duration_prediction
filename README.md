# NYC Taxi Trip Prediction

This repository contains code for predicting the duration of a taxi trip in New York City based on historical data. The project is organized into modules for data preprocessing, model training, model evaluation, and a Streamlit frontend application. The model is a Random Forest Regressor from scikit-learn. The project also includes a FastAPI backend for predictions.

## Project Structure

The project is organized as follows:

```
NYC_Taxi_Trip_Prediction/
│
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis notebook
│
├── src/
│   ├── common.py                 # Common utilities and configurations
|   ├── data_preprocessing.py     # data preprocessing script
│   ├── model_training.py         # Model training script
│   └── model_evaluation.py       # Model evaluation script
│
├── api/
│   └── api.py               # FastAPI backend for predictions
│
├── data/
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
│
├── frontend/
│   └── app.py               # Streamlit frontend application
│
├── config.yml               # YAML configuration file
├── config.ini               # INI configuration file
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```


## Data

The data used for this project is a truncated version of the [NYC Taxi Trips dataset](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) from Kaggle.

## Model

The model used for this project is a Ridge Regressor from scikit-learn. The model is trained on the following features:

* Pickup time (hour of day)
* Pickup time (day of week)
* Pickup time (month of year)

## Evaluation

The model is evaluated using mean square error (RMSE) and R-squared. The evaluation metrics are calculated using the `evaluate_model` function in `model_evaluation.py`.

## How to use

To use the model, run the following commands:

```
    $ python -m pip install -r requirements.txt
    $ python model_training.py
    $ py api/api.py
    $ streamlit run frontend/app.py

    $ uvicorn api:app --reload --host 0.0.0.0 --port 8000

```