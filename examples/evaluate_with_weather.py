import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from scripts import SoilMoisture, Trainer
from models import SoilMoisturePredictor

DEVICE = 'cuda'

checkpoints = torch.load('./saved_models/model.pt')
model = SoilMoisturePredictor().to(DEVICE)
model.load_state_dict(checkpoints)


df = pd.read_csv('./data/weather/soil_moisture_change.csv')

X = df[['Temperature', 'Precipitation']].values.astype(np.float32)
y = df[['Soil Moisture Change']].values.astype(np.float32)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X = torch.tensor(X).to(DEVICE)

model.eval()
with torch.no_grad():
    y_pred = model(X).detach().cpu().numpy()

final_y = np.concatenate((y, y_pred), axis=1)
df = pd.DataFrame(final_y, columns=['True', 'Pred'])
df.to_csv('./data/weather/predictions.csv', index=False)