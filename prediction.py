import torch
import pandas as pd
import numpy as np
from train import TransformerModel  # Ensure TransformerModel is defined in train.py

# Load trained model
model = TransformerModel(input_dim=5, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

data = pd.read_excel('data.xlsx')

# Extract input features (first 5 columns)
X = data.iloc[:, :5].values

# Generate predictions for all rows
predictions = []
with torch.no_grad():
    for row in torch.tensor(X, dtype=torch.float32):
        output = model(row.unsqueeze(0))  # Add batch dimension
        predictions.append(output.item())

# Save predictions to Excel
data['Predicted_SR'] = predictions
data.to_excel('data_with_predictions.xlsx', index=False)

print("Predictions saved successfully in 'data_with_predictions.xlsx'.")
