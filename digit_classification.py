import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import sys


np.random.seed(1)

# Loading and preparing the MNIST dataset
mnist = fetch_openml('mnist_784')

X = mnist.data.to_numpy().reshape(-1, 28, 28)
Y = mnist.target.to_numpy().astype(int)

shuffle = np.random.permutation(70000)
Xtrain, Xtest = X[shuffle[:60000]], X[shuffle[60000:]]
Ytrain, Ytest = Y[shuffle[:60000]], Y[shuffle[60000:]]

Xtrain_tensor = torch.tensor(Xtrain.reshape(-1, 1, 28, 28), dtype=torch.float32)
Xtest_tensor = torch.tensor(Xtest.reshape(-1, 1, 28, 28), dtype=torch.float32)
Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.long)
Ytest_tensor = torch.tensor(Ytest, dtype=torch.long)

train_dataset = TensorDataset(Xtrain_tensor, Ytrain_tensor)
test_dataset = TensorDataset(Xtest_tensor, Ytest_tensor)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Selecting the device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the model 1
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# Defining the model 2
class CNN6Layer(nn.Module):
    def __init__(self):
        super(CNN6Layer, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):

        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))

        x = nn.ReLU()(self.conv4(x))

        # Flattening the output
        x = x.view(-1, 256 * 3 * 3)

        # Fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

        return x

# Define the models
model = NeuralNet().to(device)
cnn_model = CNN6Layer().to(device)

# Loss function and optimizer selection - we need to change the input to optimizer while changing model selection
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.003)
training_accuracies = []

# Training function
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass and loss calculation
        pred = model(X)
        loss = loss_func(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate accuracy for this batch
        correct += (pred.argmax(1) == y).sum().item()

        if batch % 200 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
            sys.stdout.flush()

    # Calculate accuracy for the epoch
    epoch_accuracy = correct / size
    return epoch_accuracy

def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()

    accuracy = correct / size
    print(f"Accuracy: {accuracy * 100:.2f}%")


# Train with either NN or CNN by chaning model input to the function
epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch_accuracy = train(train_dataloader, cnn_model, loss_func, optimizer)
    training_accuracies.append(epoch_accuracy)

# Plotting training accuracies through epochs
plt.plot(range(1, epochs + 1), training_accuracies, marker='o')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

# Printing model archtecture and performance
print("Model Architecture and Parameters:")
summary(cnn_model, input_size=(1, 28, 28))

print("\n Model Performance on the training set:")
test(train_dataloader, cnn_model)

print("\n Model performance on the test set:")
test(test_dataloader, cnn_model)

print("Done!")
