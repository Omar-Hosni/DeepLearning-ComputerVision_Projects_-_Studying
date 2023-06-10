from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
#plt.show()


def accuracy_fn(y_true, y_predi):
    correct = torch.eq(y_true, y_predi).sum().item()
    return (correct / len(y_predi)) * 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.sequential_layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, output_features),

            #add nn.ReLU activation functions if requires non-linear layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential_layers(x)

model = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8).to(device)
X_train = X_train.to(device)  # Move input data to GPU
y_train = y_train.to(device)  # Move input data to GPU

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 1000
torch.manual_seed(42)

# Make prediction logits with model
y_logits = model(X_test.to(device))

# Perform softmax calculation on logits across dimension 1 to get prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

'''
CrossEntropyLoss() expects integer labels as targets, 
so it's important to convert the target values to torch.
long before passing them to the loss function.
'''
y_train, y_test = y_train.long(), y_test.long()

for i in range(epochs):
    model.train()

    y_logits = model(X_train)  # model outputs raw logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # go from logits -> prediction probabilities -> prediction labels

    # print(y_logits)
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)
    #acc = accuracy_fn(y_true=y_train, y_predi=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        test_logits = model(X_test)  # Move input data to GPU
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_acc = accuracy_fn(y_true=y_test, y_predi=test_pred)

        test_loss = loss_fn(y_test, test_logits)  # Move input data to GPU

        if i % 100 == 0:
            print(f"Epoch: {i} | Loss: {loss:.5f} | Test Loss: {test_logits}")
