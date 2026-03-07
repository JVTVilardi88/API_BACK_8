from typing import List
from pydantic import BaseModel


class CrimeInput(BaseModel):
    fecha: str
    hora: int
    latitud: float
    longitud: float
    distrito: str


class MultipleDataInputs(BaseModel):
    inputs: List[CrimeInput]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "fecha": "2026-03-07",
                        "hora": 12,
                        "latitud": 37.777635,
                        "longitud": -122.434720,
                        "distrito": "NORTHERN"
                    }
                ]
            }
        }


class PredictionResults(BaseModel):
    crimen_predicho: str
    probabilidad: float
    latency_ms: float
    note: str

    class Config:
        schema_extra = {
            "example": {
                "crimen_predicho": "LARCENY/THEFT",
                "probabilidad": 0.2485,
                "latency_ms": 23.4,
                "note": "Resultado generado por API XGBoost"
            }
        }