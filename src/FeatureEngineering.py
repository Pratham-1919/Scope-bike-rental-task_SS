import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateExtractoer(BaseEstimator, TransformerMixin):
    def __init__(self, DATE: str = "Date"):
        self.DATE = DATE

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X[self.DATE]):
            X[self.DATE] = pd.to_datetime(X[self.DATE], format="%d/%m/%Y")

        X["Day"] = X[self.DATE].dt.day
        X["Month"] = X[self.DATE].dt.month
        X["day_of_week"] = X[self.DATE].dt.dayofweek
        return X.drop(columns=[self.DATE])
    

class FlagFeatureAdder(BaseEstimator, TransformerMixin):
       
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)
        X["is_holiday"] = (X["Holiday"] == "Holiday").astype(int)
        X["non_working_day"] = ((X["is_weekend"] == 1) | (X["is_holiday"] == 1)).astype(int)
        X["is_raining"] = (X["Rainfall(mm)"] > 0).astype(int)
        X["is_snowing"] = (X["Snowfall (cm)"] > 0).astype(int)
        X["is_rush_hour"] = X["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        return X.drop(columns=["Holiday"])
    


    

class DropColumns(BaseEstimator, TransformerMixin):
       
    DEFAULT_DROP_COLS = [
        "Dew point temperature_C",
        "Functioning Day",
        "Year",
        "day_name",
    ]

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns if drop_columns is not None else self.DEFAULT_DROP_COLS
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        return X.drop(columns=self.drop_columns, errors="ignore")