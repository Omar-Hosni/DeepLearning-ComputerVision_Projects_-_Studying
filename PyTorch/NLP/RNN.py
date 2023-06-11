import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

seq_len = 20
time_steps = np.linspace(0, np.pi, seq_len+1)
data = np.sin(time_steps)
data.resize((seq_len+1, 1))

X = data[:-1]
y = data[1:]

#plt.plot(time_steps[1:], X, 'r.', label='input,x')
#plt.plot(time_steps[1:], y, 'b.', label='input,x')
#plt.legend(loc='best')
#plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return output, hidden


#Training
input_size=1
output_size=1
hidden_dim=32
n_layers=1

rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

def train(rnn, n_steps, print_every):
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_len+1)
        data = np.sin(time_steps)
        data.resize((seq_len+1, 1))

        X = data[:-1]
        y = data[1:]

        X_tensor = torch.Tensor(X).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        prediction, hidden = rnn(X_tensor, hidden)
        hidden = hidden.data

        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            #plt.plot(time_steps[1:], x,'r.') #Input
            #plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.') #prediction
            #plt.show()

        return rnn

n_steps= 75
print_every=15
trained_rnn=train(rnn, n_steps, print_every)
print(trained_rnn)
