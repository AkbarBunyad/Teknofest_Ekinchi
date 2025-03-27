import numpy as np
import json

class SoilMoisturePredictor:
    def __init__(self, weights_path):
        # Load weights from JSON file
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        
        # Initialize weights and biases from loaded data
        self.linear_1_weight = np.array(weights['linear_1.weight'])  # (256, 2)
        self.linear_1_bias = np.array(weights['linear_1.bias'])      # (256,)
        self.linear_2_weight = np.array(weights['linear_2.weight'])  # (64, 256)
        self.linear_2_bias = np.array(weights['linear_2.bias'])      # (64,)
        self.linear_3_weight = np.array(weights['linear_3.weight'])  # (16, 64)
        self.linear_3_bias = np.array(weights['linear_3.bias'])      # (16,)
        self.linear_4_weight = np.array(weights['linear_4.weight'])  # (1, 16)
        self.linear_4_bias = np.array(weights['linear_4.bias'])      # (1,)

    def __call__(self, x):
        return self.forward(x)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def linear(self, x, weight, bias):
        """Linear layer computation"""
        return np.dot(x, weight.T) + bias

    def forward(self, x):
        """Forward pass through the network"""
        # Ensure input is numpy array
        x = np.array(x).astype(np.float32)
        
        # Layer 1
        x = self.linear(x, self.linear_1_weight, self.linear_1_bias)
        x = self.relu(x)
        
        # Layer 2
        x = self.linear(x, self.linear_2_weight, self.linear_2_bias)
        x = self.relu(x)
        
        # Layer 3
        x = self.linear(x, self.linear_3_weight, self.linear_3_bias)
        x = self.relu(x)
        
        # Layer 4 (output layer)
        x = self.linear(x, self.linear_4_weight, self.linear_4_bias)
        
        return x

# Example usage:
# model = SoilMoisturePredictor('path/to/model_weights.json')
# prediction = model.forward(np.array([[input1, input2]]))