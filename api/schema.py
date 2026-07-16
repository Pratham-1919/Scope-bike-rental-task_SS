from pydantic import BaseModel, Field
from typing import Literal

class BikePredictionInput(BaseModel):
    Date: str = Field(..., example="14/07/2026", description="Format: DD/MM/YYYY")
    Hour: int = Field(..., ge=0, le=23, example=18)
    Temperature_C: float = Field(..., alias="Temperature_C", example=25.4)
    Humidity: int = Field(..., alias="Humidity(%)", ge=0, le=100, example=55)
    Wind_speed: float = Field(..., alias="Wind speed (m/s)", ge=0.0, example=1.5)
    Visibility: int = Field(..., alias="Visibility (10m)", ge=0, example=2000)
    Dew_point: float = Field(..., alias="Dew point temperature_C", example=15.0)
    Solar_Radiation: float = Field(..., alias="Solar Radiation (MJ/m2)", ge=0.0, example=0.0)
    Rainfall: float = Field(..., alias="Rainfall(mm)", ge=0.0, example=0.0)
    Snowfall: float = Field(..., alias="Snowfall (cm)", ge=0.0, example=0.0)
    Seasons: Literal["Winter", "Spring", "Summer", "Autumn"] = Field(..., example="Summer")
    Holiday: Literal["Holiday", "No Holiday"] = Field(..., example="No Holiday")

    class Config:
        populate_by_name = True

class PredictionOutput(BaseModel):
    predicted_bike_count: int
    model_version: str
    timestamp: str