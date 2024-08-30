from torch import nn

class SoilMoisturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(2, 256)
        self.linear_2 = nn.Linear(256, 64)
        self.linear_3 = nn.Linear(64, 16)
        self.linear_4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.relu(self.linear_3(x))
        x = self.linear_4(x)
        return x