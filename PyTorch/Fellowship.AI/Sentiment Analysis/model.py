import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, classification_report
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = os.path.join("cache", "sentiment_analysis")

def read_imdb_data_and_clean():
    movie_reviews = pd.read_csv('IMDB Dataset.csv')
    movie_reviews.review = movie_reviews.review.str.lower()
    movie_reviews.review = movie_reviews.review.str.replace(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", regex=True)

    return movie_reviews


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from string import punctuation, digits

def convert_review_to_words(review):
    text = BeautifulSoup(review, 'html.parser').get_text()
    text = text.lower()
    all_text = ''.join([c for c in text if c not in punctuation and digits])
    all_text = text.replace(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "")
    words = all_text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    words = [PorterStemmer().stem(w) for w in words]
    return words


movie_reviews = read_imdb_data_and_clean()
movie_reviews.sentiment = movie_reviews.sentiment.map({'positive':1, 'negative':0})
X = movie_reviews['review'].tolist()
y = movie_reviews['sentiment'].tolist()

#
for idx, rev in enumerate(X):
    X[idx] = convert_review_to_words(X[idx])

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234, stratify=y)


#vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


#Data Loaders and Batching
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

train_dataset = Dataset(torch.tensor(X_train_bow), torch.tensor(y_train)).to(device)
test_dataset = Dataset(torch.tensor(X_test_bow), torch.tensor(y_test)).to(device)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True).to(device)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False).to(device)


#build model
import torch.nn as nn

def accuracy(outputs, labels):
    predicted = torch.round(torch.sigmoid(outputs))
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size,  n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

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

        if (torch.cuda.is_available()):
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
            )
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden



#Instantiate the network

vocab_size = vocab_size = X_train.shape[1]
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 3

model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers).to(device)
lr = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 10
batch_size = 64
clip_threshold=5

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for batch_rev, batch_label in train_loader:
        # Move data to GPU
        batch_reviews = batch_reviews.to(device)
        batch_labels = batch_labels.to(device)

        #Forward pass
        outputs = model(batch_rev)
        loss = criterion(outputs, batch_labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, batch_labels)
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    # Evaluate the model on the test set
model.eval()
test_loss = 0.0
test_acc = 0.0
num_epochs=10

with torch.no_grad():
    for batch_reviews, batch_labels in test_loader:
        # Move data to GPU
        batch_reviews = batch_reviews.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(batch_reviews)
        loss = criterion(outputs, batch_labels)

        test_loss += loss.item()
        test_acc += accuracy(outputs, batch_labels)

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = test_acc / len(test_loader)

# Print epoch statistics
print(
    f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}')


sentence = "This movie is amazing!"

processed_sentence = convert_review_to_words(sentence)
processed_sentence = CountVectorizer().fit_transform([processed_sentence]).toarray()
processed_sentence = torch.tensor(processed_sentence).float().to(device)

model.eval()
with torch.no_grad():
    output = model(processed_sentence)
    prediction = torch.sigmoid(output).item()

print("Prediction:", prediction)