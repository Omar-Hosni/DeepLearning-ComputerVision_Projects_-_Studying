import torch
import torchvision

from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


INPUT_SHAPE = 784 #because this is how many features we got, which is 28x28 pixels=784
HIDDEN_UNITS = 8
OUTPUT_SHAPE = len(train_data.classes)

model = FashionMNISTModelV0(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=OUTPUT_SHAPE)
print(torch.cuda.is_available())
model.to('cpu')

from helper_functions import accuracy_fn

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Import tqdm for progress bar
from tqdm.auto import tqdm
from timeit import default_timer as timer

def print_trian_time(start: float, end: float, device: torch.device = None):
    time_taken = end-start
    print(f'Train time on:{device}: {end-start:.3f} seconds')
    return time_taken

torch.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3

for i in tqdm(range(epochs)):
    print(f'Epoch: {i}\n------')

    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss, test_acc = 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in test_dataloader:
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            train_loss += loss
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")


train_time_end_on_cpu = timer()

total_train_time = print_trian_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=str(next(model.parameters()).device))

print(total_train_time)

torch.manual.seed(42)
def eval_model(model, data_loader, loss_fn, acc_fn):
    loss, acc = 0,0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

model_res = eval_model(model=model, data_loader=test_dataloader, loss_fn=loss_fn, acc_fn=accuracy_fn())



