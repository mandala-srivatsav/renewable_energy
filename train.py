import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
X = data[['WS', 'RH', 'TAVG', 'TMAX', 'TMIN']].values
y = data['SR'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define dataset
class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
dataset = SolarDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

# Initialize Model
model = TransformerModel(input_dim=5, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully as 'model.pth'.")
