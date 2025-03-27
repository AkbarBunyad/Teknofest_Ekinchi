""""
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
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from typing import List

def predict_soil_moisture(model, temperature: List, precipitation: List) -> np.ndarray:
    # Convert input lists to numpy array and transpose
    sample = np.array([temperature, precipitation], dtype=np.float32).T
    
    # Pre-defined scaling parameters
    mins = np.array([[1.2, 0.0]], dtype=np.float32)
    maxs = np.array([[41.0, 22.7]], dtype=np.float32)

    # Min-max scaling
    sample_scaled = (sample - mins) / (maxs - mins)

    # Forward pass through the model
    soil_moisture_change = model.forward(sample_scaled)
    
    return soil_moisture_change