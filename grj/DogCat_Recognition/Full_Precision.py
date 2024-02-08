import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data_Read import DogCatData
import matplotlib.pyplot as plt

# Setting KMP_DUPLICATE_LIB_OK environment variable (Use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

label_dict = {"Cat": 0, "Dog": 1}
# Some training constants
learning_rate = 2.5e-3
batch_size = 30
epochs = 100
accuracy = torch.zeros(epochs)

# import training data and test data
training_dataset = DogCatData(".\\Training_Data\\")
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DogCatData(".\\Test_Data\\")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# define working device
device = "cuda" if torch.cuda.is_available() else "cpu"


# fabricate neural network
class FullPrecisionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.normalization1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.normalization2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.normalization3 = nn.BatchNorm2d(num_features=64)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=16384, out_features=2)

        self.stack = nn.Sequential(
            self.conv1,
            self.activation,
            self.normalization1,
            self.pooling,

            self.conv2,
            self.activation,
            self.normalization2,
            self.pooling,

            self.conv3,
            self.activation,
            self.normalization3,
            self.pooling,

            self.flatten,
            self.dense,
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        logit = self.stack(x)
        return logit


model = FullPrecisionCNN().to(device)  # Move model to the chosen device


def TrainLoop(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    model.train()  # Set model in training mode

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Move data to the device
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def TestLoop(dataloader, model, loss_fn, t):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()  # Set model in evaluation mode
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move data to the device
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # accuracy calculation

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    accuracy[t] = correct


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):  # Correct 8: Typo corrected in variable name
        print(f"Epoch {t + 1}\n-------------------------------")
        TrainLoop(training_dataloader, model, loss_fn, optimizer, t)
        TestLoop(test_dataloader, model, loss_fn, t)
    print("Done!")

    torch.save(model.state_dict(), 'full_precision_weights.pth')

    plt.plot(torch.arange(1, epochs + 1), accuracy, 'b')
    plt.show()
