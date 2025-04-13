import fastapi
import pandas as pd
from challenge.utils import FEATURES_COLS, load_the_pipe, get_feature_names
from challenge.model import DelayModel
from challenge.schemas import FlightsData
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
app = fastapi.FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can also use ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],                      # Or be specific: ["POST", "GET"]
    allow_headers=["*"],                      # Or be specific: ["Content-Type", "Authorization"]
)

@app.options("/predict")
async def options_predict(request: Request):
    return JSONResponse(
        status_code=200,
        content={"message": "CORS preflight OK"},
        headers={
            "Access-Control-Allow-Origin": "*",  # or specify http://localhost:5173
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: FlightsData) -> dict:
    model = DelayModel()  # Instantiate the model
    
    # Convert input flights to a DataFrame
    flights = [flight.dict() for flight in data.flights]
    input_df = pd.DataFrame(flights)
    
    try:
        # Try to load and use the preprocessing pipeline
        pipeline = load_the_pipe()  # Uses default path
        
        # Add dummy date columns required by the transformer
        dummy_date = '2023-01-01 00:00:00'
        input_df['Fecha-I'] = dummy_date
        input_df['Fecha-O'] = dummy_date
        
        # Transform data
        transformed_data = pipeline.transform(input_df)
        
        # Create DataFrame with feature names
        features_df = pd.DataFrame(transformed_data, columns=get_feature_names(pipeline))
        
        # Extract only the features needed for prediction
        features = features_df[FEATURES_COLS].astype(int)
        
    except Exception as e:
        # Fallback to manual feature creation if pipeline loading fails
        print(f"Failed to use pipeline: {e}. Falling back to manual feature creation.")
        
        opera_columns = [col for col in FEATURES_COLS if "OPERA" in col]
        type_columns = [col for col in FEATURES_COLS if "TIPOVUELO" in col]
        month_columns = [col for col in FEATURES_COLS if "MES" in col]
        
        feature_list = []
        for flight in flights:
            dict_data = {}
            
            airline = flight["OPERA"]
            type_ = flight["TIPOVUELO"]
            month = flight["MES"]
            
            for col in FEATURES_COLS:
                if col in opera_columns:
                    dict_data[col] = 1 if airline in col else 0
                elif col in type_columns:
                    dict_data[col] = 1 if type_ in col else 0
                else:
                    dict_data[col] = 1 if str(month) in col else 0
                    
            feature_list.append(dict_data)
            
        features = pd.DataFrame(feature_list)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Return the response
    return {"predict": list(predictions)}
