import numpy as np

#load data
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt','r') as f:
    labels = f.read()

#let's remove all punctuation
from string import punctuation

reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])

#split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

#create a list of words
words = all_text.split()

from collections import Counter

#build a dictionary that maps a words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


#use the dict to tokenize each review in review_splits
#store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in reviews.split()])


#encoding the labels
#1 positive, 0 negative
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

#outliers review
reviews_lens = Counter([len(x) for x in reviews_ints])
print(f'Zero-length reviews: {reviews_lens[0]}')
print(f'Maximum-length reviews: {max(reviews_lens)}')

#remove any review or labels with zero length from the review_ints list
non_zero_len_idx = [i for i, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[i] for i in non_zero_len_idx]
encoded_labels = np.array([encoded_labels[i] for i in non_zero_len_idx])

#padding sequences to deal with both and very long reviews, we'll pad or truncate all our reviews to a specific length

def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

#test
seq_length = 200
features = pad_features(reviews_ints, seq_length)
print(features)

#training, validation, test
split_frac = 0.8

split_idx = int(len(features)*0.8)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]



#Data Loaders and Batching
import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, batch_size=50)
valid_loader = DataLoader(valid_data, batch_size=50)
test_loader = DataLoader(test_data, batch_size=50)

#obtain one batch of training data
data_iter = iter(train_loader)
sample_x, sample_y = data_iter.next()

#Build model
train_on_gpu = torch.cuda.is_available()

import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers,  drop_prob=0.5):
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embed(x)

        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dum)

        sig_out = self.sigmoid(self.fc(self.dropout(lstm_out)))
        sig_out = sig_out.view(batch_size, -1)

        return sig_out[:, -1], hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
            )


        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden



#Instantiate the network

vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 3

model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 4
batch_size=50


counter = 0
clip_threshold = 5 #gradient clipping

if train_on_gpu:
    model.cuda()

for e in range(epochs):
    h = model.init_hidden(batch_size)

    #loop through the data set
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([hidden_layer.data for hidden_layer in h])

        #zero accumulated gradients
        model.zero_grad()

        #get the output from the model
        output, h = model(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
        optimizer.step()

        #loss stats
        if counter % 10 == 0:
            val_hid = model.init_hidden(batch_size)
            val_losses = []
            model.eval()

            for inputs, labels in valid_loader:
                val_h = tuple([hidden_layer.data for hidden_layer in h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, valh_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))




#Testing..
test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)
model.eval()

for inputs, labels in test_loader:
    h = tuple([hidden_layer.data for hidden_layer in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    outputs, h = model(inputs, h)

    loss = criterion(outputs.squeeze(), labels.float())
    loss.backward()
    test_losses.append(loss)

    #convert output probabilities to predicted class (0 or 1)
    pred = torch.round(outputs.squeeze())

    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))