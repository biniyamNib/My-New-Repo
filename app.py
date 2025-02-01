from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('movie_rating_model.pkl')

app = FastAPI()

# Define the input data structure
class MovieRatingInput(BaseModel):
    userId: float
    movieId: float
    genres_encoded: float
    rating: float

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Rating Prediction API!"}

# Define the prediction endpoint
@app.post("/predict/")
def predict(input_data: MovieRatingInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"prediction": bool(prediction[0])}
