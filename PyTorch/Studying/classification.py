from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


'''
We'll use the make_circles() method from Scikit-Learn 
to generate two circles with different coloured dots.
'''

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
circles.head(10)

plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
#plt.show()

#turn data into tensors
#otherwise this causes issues with computations
import torch
from torch import nn

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#build a model

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV0(nn.Module):
    def __init__(self):
        super(CircleModelV0, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        return x


model = CircleModelV0().to(device)
print(model)


#replicate above CircleModelV0 but using nn.Sequential, just like tensorflow.keras

model = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1),
).to(device)


loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

'''
Accuracy can be measured by dividing 
the total number of correct predictions 
over the total number of predictions.
'''

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct/len(y_pred))*100
    return acc


# View the frist 5 outputs of the forward pass on the test data
y_logits = model(X_test.to(device))[:5]

'''
prediction probabilites, is the values generated based on 
how much the model things the data point belongs to one 
class or another
'''

# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)

#find the predicted labels(round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

y_pred_labels = torch.round(torch.sigmoid(model(X_test.to(device))[:5]))

print(torch.eq(y_pred_labels.squeeze(), y_preds.squeeze()))

y_preds.squeeze() #get rid of extra dimensions


#train and test

torch.manual_seed(42)
epochs = 1000

#put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for i in range(epochs):
    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if i % 100 == 0:
        print(f'Epoch: {i}, Loss:{loss}, Accuracy: {test_acc}, Test Loss: {test_loss}')

