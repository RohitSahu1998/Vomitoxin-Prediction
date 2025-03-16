from fastapi import FastAPI, HTTPException, Request
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel
import uvicorn
import logging
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Define the Optimized Neural Network Model
class FinalOptimizedNN(nn.Module):
    def __init__(self, input_dim):
        super(FinalOptimizedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Step 2: Load the Trained Model
device = torch.device("cpu")
input_dim = 449  # Adjust based on dataset

# Initialize & Load Model
model = FinalOptimizedNN(input_dim).to(device)
model.load_state_dict(torch.load(r"C:\Users\umapa\Desktop\tasc\final_optimized_model.pth"))
model.eval()

# Step 3: Initialize FastAPI App
app = FastAPI(title="Vomitoxin Prediction API with Tracing",
              description="Predict vomitoxin levels from spectral data with detailed logging.")

# Step 4: Define Input Schema
class SpectralInput(BaseModel):
    spectral_values: list  # Expecting a list of reflectance values

# Step 5: Middleware for Logging Request Details
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log Incoming Request
    logging.info(f"📩 Incoming request: {request.method} {request.url}")
    
    # Process the Request
    response = await call_next(request)
    
    # Calculate Execution Time
    process_time = time.time() - start_time
    logging.info(f"✅ Completed request in {process_time:.4f} seconds")
    
    return response

# Step 6: Prediction Endpoint with Tracing
@app.post("/predict")
def predict(data: SpectralInput):
    logging.info(f"🔍 Received input: {data.spectral_values[:5]}... ({len(data.spectral_values)} features)")

    try:
        # Convert input to tensor
        spectral_values = torch.tensor(data.spectral_values, dtype=torch.float32).view(1, -1).to(device)

        # Validate Feature Size
        if spectral_values.shape[1] != input_dim:
            logging.error(f"❌ Invalid input size! Expected {input_dim}, but got {spectral_values.shape[1]}")
            raise HTTPException(status_code=400, detail=f"Expected {input_dim} spectral features, but got {spectral_values.shape[1]}")

        # Make Prediction
        with torch.no_grad():
            prediction = model(spectral_values).item()
        
        # Log Prediction
        logging.info(f"🎯 Prediction: {prediction:.2f} vomitoxin_ppb")

        return {"predicted_vomitoxin_ppb": round(prediction, 2)}
    
    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Step 7: Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
