from pydantic import BaseModel, Field, validator
from typing import List
from fastapi import HTTPException


# Define the request and response models
class FlightData(BaseModel):
    OPERA: str = Field(..., description="Name of the airline that operates")
    TIPOVUELO: str = Field(..., description="Type of flight: 'I' for International, 'N' for National")
    MES: int = Field(..., description="Month number of the flight operation (1-12)")

    @validator("OPERA")
    def validate_opera(cls, value):
        allowed_airlines = {'American Airlines', 'Air Canada', 'Air France', 'Aeromexico',
                            'Aerolineas Argentinas', 'Austral', 'Avianca', 'Alitalia',
                            'British Airways', 'Copa Air', 'Delta Air', 'Gol Trans', 'Iberia',
                            'K.L.M.', 'Qantas Airways', 'United Airlines', 'Grupo LATAM',
                            'Sky Airline', 'Latin American Wings', 'Plus Ultra Lineas Aereas',
                            'JetSmart SPA', 'Oceanair Linhas Aereas', 'Lacsa'}
        if value not in allowed_airlines:
            raise HTTPException(status_code=400, detail=f"Invalid airline: {value}. Must be one of {allowed_airlines}.")
        return value

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value):
        allowed_types = {"N", "I"}
        if value not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Invalid type: {value}. Must be one of {allowed_types}.")
        return value

    @validator("MES")
    def validate_month(cls, value):
        allowed_months = set(range(1, 13))
        if value not in allowed_months:
            raise HTTPException(status_code=400, detail=f"Invalid month: {value}. Must be one of {allowed_months}.")
        return value


class FlightsData(BaseModel):
    flights: List[FlightData]


class PredictionResponse(BaseModel):
    predict: List[int]