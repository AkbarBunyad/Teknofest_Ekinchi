import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import MSELoss

from scripts import SoilMoisture, Trainer
from models import SoilMoisturePredictor

DEVICE = 'cuda'

df = pd.read_csv('./data/weather/soil_moisture_change.csv')

df_min = df.min()[['Temperature', 'Precipitation']].values.reshape(-1, 2).astype(np.float32)
df_max = df.max()[['Temperature', 'Precipitation']].values.reshape(-1, 2).astype(np.float32)

X = df[['Temperature', 'Precipitation']].values.astype(np.float32)
y = df[['Soil Moisture Change']].values.astype(np.float32)

model = SoilMoisturePredictor().to(DEVICE)

dataset = SoilMoisture(X=X, y=y, x_min=df_min, x_max=df_max)
loader = DataLoader(dataset, batch_size=4)

loss_fn = MSELoss()
optimizer = SGD(params=model.parameters(), lr=0.001)

trainer = Trainer(model=model, train_loader=loader, val_loader=loader, optimizer=optimizer, 
        loss_fn=loss_fn, epochs=500, filepath='./saved_models/model.pt', device=DEVICE)

trainer.run()