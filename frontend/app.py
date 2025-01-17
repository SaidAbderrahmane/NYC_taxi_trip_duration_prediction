import streamlit as st
import requests

# Set up the title
st.title("NYC Taxi Trip Prediction")

# Get user input
hour = st.slider("Hour", 0, 23, 12)
weekday = st.slider("Weekday", 0, 6, 0)
month = st.slider("Month", 1, 12, 1)

# Make prediction button
if st.button("Predict"):
    # Send data to the FastAPI backend
    response = requests.post(
        "http://localhost:8000/predict",
        json={"hour": hour, "weekday": weekday, "month": month}
    )
    
    if response.status_code == 200:
        predictions = response.json()
        st.write("Prediction: ", predictions[0], "minutes")
    else:
        st.write("Error: Unable to get prediction")
