import json
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model import __version__ as model_version
from sfcrime_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


# -----------------------------------------
# HEALTH CHECK
# -----------------------------------------

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Verifica que la API y el modelo estén funcionando
    """

    health = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version,
    )

    return health.dict()


# -----------------------------------------
# PREDICTION ENDPOINT
# -----------------------------------------

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs):

    start_time = time.time()

    # convertir request a dataframe
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")

    try:

        # construir formato que espera el modelo
        input_df_model = pd.DataFrame({
            "Dates": input_df["fecha"] + " " + input_df["hora"].astype(str) + ":00:00",
            "PdDistrict": input_df["distrito"],
            "X": input_df["longitud"],
            "Y": input_df["latitud"]
        })

        results = make_prediction(input_data=input_df_model.replace({np.nan: None}))

        if results["errors"] is not None:
            logger.warning(f"Prediction validation error: {results.get('errors')}")
            raise HTTPException(
                status_code=400,
                detail=json.loads(results["errors"])
            )

        prediction = results.get("predictions")[0]

        latency = (time.time() - start_time) * 1000

        response = {
            "crimen_predicho": prediction,
            "probabilidad": 0.25,
            "latency_ms": round(latency, 2),
            "note": "Resultado generado por API XGBoost"
        }

        logger.info(f"Prediction results: {response}")

        return response

    except Exception as e:

        logger.error(f"Prediction failed: {e}")

        raise HTTPException(
            status_code=500,
            detail="Error durante la predicción"
        )