import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


with open('textfile.txt','r') as f:
    text = f.read()

print('cuda: ', torch.cuda.is_available())

#tokenization
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch : ii for ii, ch in int2char.items()}

#encode
encoded = np.array([char2int[ch] for ch in text])

def one_hot_encode(arr, n_labels):
    #initialize the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    #fill the appropriate
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1

    #finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

test_seq = np.array([[3,5,1]])
one_hot = one_hot_encode(test_seq, 8)

print(one_hot)

def get_batches(arr, batch_size, seq_len):
    '''
    create a generator that returns batches of size
    batch_size x seq_len from arr
    Args:
        arr: array you want to make batches from
        batch_size: batch size, the number of sequences per batch
        seq_len: number of encoded chars in sequence
    '''

    ## TODO: get the number of batches we can make
    n_batches = len(arr) // (batch_size * seq_len)

    ## TODO: Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size * seq_len]

    ## TODO: Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], seq_len):
        # the features
        x = arr[:, n:n + seq_len]
        # the targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_len]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y



batches = get_batches(encoded, 8, 50)
try:
    while True:
        x, y = next(batches)
        # Do something with each batch (x and y)
        print("Batch:", x.shape, y.shape)
except StopIteration:
    pass

#check if GPU is available
train_on_gpu = torch.cuda.is_available()

class charRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch : ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fully_connected = nn.Linear(n_hidden, len(self.chars))


    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.view(-1, self.n_hidden)
        out = self.fully_connected(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

            return hidden


def train(net, data, epochs=10, batch_size=10, seq_len=50, lr=0.001, clip=5, val_frac=0.3, print_every=2):

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)

    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_len):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                input, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net(inputs,h)

            loss = criterion(output, targets.view(batches * seq_len))
            loss.backward()

            #clip gradients cause it could get big, clip when it crosses the threshold(clip)
            nn.util.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                for x, y in get_batches(val_data, batch_size, seq_len):
                    x = one_hot_encode(x, n_chars)
                    x,y = torch.from_numpy(x), torch.from_numpy(y)
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x,y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_len))

                    val_losses.append(val_loss)

                print(f'Epoch {e+1}/{epochs}')
                print(f'step {counter}')
                print(f'loss {loss.item()}')
                print(f'val loss: {np.mean(val_losses)}')


n_hidden = 512
n_layers = 2
net = charRNN(chars, n_hidden, n_layers)
print(net)

batch_size=128
seq_len = 100
n_epochs=20
print(train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_len=seq_len, lr=0.001))

import torch.nn.functional as F

def predict(net, char, h=None, top_k=None):
    'given a character predict the next character'

    x=np.array([[net.char2int[char]]])
    x=one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if(train_on_gpu):
        inputs = inputs.cuda()

    h = tuple([each.data for each in h])
    out,h = net(inputs, h)

    p = F.softmax(out, dim=1).data
    if(train_on_gpu):
        p = p.cpu()

    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    return net.int2char[char], h

def sample(net, size, prime='The', top_k=None):
    if(train_on_gpu):
        net.cuda()

    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
        chars.append(char)

    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)

print(sample(net, 1000, prime='Becuase', top_k=5))

