import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = {
    'age': [25, 30, 22, 35, 45],
    'income': [50000, 60000, 40000, 75000, 90000],
    'education': ['Bachelors', 'Masters', 'High School', 'PhD', 'Masters']
}
df = pd.DataFrame(data)

#1. Binning/Numerical Encoding
bins = [0, 25, 30, 35, np.inf]
labels = ['<25', '25-30', '30-40', '40+']
df['age_group'] = pd.cut(df['age'], bins, labels=labels)

#2. One-Hot Encoding
df = pd.get_dummies(df, columns=['education'], prefix='edu')

#3. Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['income_scaled'] = scaler.fit_transform(df['income'])

#4. Text Features - Extracting Length
df['edu_length'] = df['education'].apply(len)

#5. Date Features - Extracting Year and Month
df['year'] = pd.to_datetime('today').year
df['month'] = pd.to_datetime('today').month

#6. Interaction Features
df['age_income_interaction'] = df['age'] * df['income']

#7. Aggregation
education_grouped = df.groupby('education')['income'].mean()
education_grouped.rename(columns={'income':'avg_income_by_edu'}, inplace=True)
df = pd.merge(df, education_grouped, on='education', how='left')

#8. Time Since Event
df['days_since_last_purchase'] = (pd.to_datetime('today') - pd.to_datetime('2023-01-01')).dt.days

#9. Boolean Features
df['high_income'] = df['income'] > df['income'].mean()

#10. Feature Crosses
df['age_group_education'] = df['age_group'] + '_' + df['education']

print(df)


# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Split data into features and target
X = df[['age', 'age_group', 'education']]
y = df['income']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
input_size = X_train_tensor.shape[1]
model = Net(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print("Test Loss:", test_loss.item())
