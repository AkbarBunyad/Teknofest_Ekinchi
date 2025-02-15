import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from typing import List

def predict_soil_moisture(model, temperature: List, precipitation: List) -> np.ndarray:
    sample = torch.tensor([temperature, precipitation]).to(torch.float32).T
    
    # Pre-defined scaling parameters
    mins = torch.tensor([1.2, 0.0]).view(1, 2).to(torch.float32)
    maxs = torch.tensor([41.0, 22.7]).view(1, 2).to(torch.float32)

    sample_scaled = (sample - mins) / (maxs - mins)

    model.eval()
    with torch.no_grad():
        soil_moisture_change = model(sample_scaled).numpy()
    return soil_moisture_change
