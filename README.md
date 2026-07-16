# Seoul Bike Rental Demand Prediction

This is a rental bike prediction dataset where we need to predict the count to bike which we need to predict based on the weather condition, season and hours(peak hours)



# Features in it

It contains the following featues in it.
    - **Date**
    - **Rented Bike Count**
    - **Hour**
    - **Temperature(ï¿½C)**
    - **Humidity(%)**
    - **Wind speed (m/s)**
    - **Visibility (10m)**
    - **Dew point temperature(ï¿½C)**
    - **Solar Radiation (MJ/m2)**
    - **Rainfall(mm)**
    - **Snowfall (cm)**
    - **Seasons**
    - **Holiday**
    - **Functioning Day**



# For Training the model

python -m src.train
This generates a freshly serialized .pkl package inside models/ containing the exact custom feature transformer references.

# Launching the Live API Server

Spin up the Uvicorn web wrapper to host the FastAPI inference pipeline: python -m uvicorn api.main:app --reload
