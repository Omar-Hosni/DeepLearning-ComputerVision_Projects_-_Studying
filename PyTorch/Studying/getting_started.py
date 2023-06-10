import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10])
print(y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train))
print(len(y_train))

#plt.scatter(X_train, y_train)
#plt.show()

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})
    plt.show()

#plot_predictions()


#Linear Regression
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        #init model parameters
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))


    #forward propagation, computation performed at every call
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()))

y_preds = None
with torch.no_grad():
    y_preds = model_0(X_test)
    print('predictions:')
    print(y_preds)

print((y_preds-y_test))

loss_fn = nn.L1Loss() #MAE
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

#steps to train
'''
1-forward pass: model(x_train)
2-calc loss: loss = loss_fn(y_pred,y_train)
3-optimize zero grad: optimizer.zero_grad()
4-loss backwards: loss.backwards
'''



torch.manual_seed(42)
epochs = 100

#setup empty lists to keep track of model progress
epoch_count = []
train_loss_values = []
test_loss_values = []


for i in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # now testing steps
    '''
    torch no grad: model(x_test)
    forward pass: loss = loss_fn(y_pred, y_test)
    calc loss: custom functions
    '''

    model_0.eval()

    with torch.no_grad():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if i % 10 == 0:
        epoch_count.append(i)
        train_loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f'Epoch: {i}| MAE Train Loss: {loss} | MAE Test Loss: {test_loss}')

#saving the model
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "getting_started.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f'Saving model to :{MODEL_SAVE_PATH}')
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

#to load the state
f='models/getting_started.pth'
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f))

#to test the loaded model
loaded_model.eval()

loaded_model_preds = None
with torch.no_grad():
    loaded_model_preds = loaded_model(X_test)

print(y_preds == loaded_model_preds)


#SUMMING EVERYTHING UP

#creating another linear regression
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.linear_layer(x)

#init
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1.state_dict())

#Training

loss_fn = nn.L1Loss() #MAE
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 100

for i in range(epochs):
    model_1.train()

    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    #test
    model_1.eval()

    with torch.no_grad():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if i % 10 == 0:
        print(f'Epoch: {i}, Train Loss: {loss}, Test loss: {test_loss}')

        




