import os  # Setting KMP_DUPLICATE_LIB_OK environment variable (Use with caution)

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from BinaryFunctions import BinaryReLU
from Layers import BinaryLinear

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Some training constants
learning_rate = 5e-4
batch_size = 30
kk_list = [1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,10,
           15,15,15,15,20,20,20,20,20,30,30,30,30,30,50,50,50,50,50,75,75,75,75,75,100,100,100,100,100,
           150,150,150,150,150,200,200,200,200,200,300,300,300,300,300,500,500,500,500,500,999,999,999,999,999]
epochs = len(kk_list)
accuracy = torch.zeros(epochs)

# Training data and test data
training_data = datasets.MNIST(
    root='data_train',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data_test',
    train=False,
    download=True,
    transform=ToTensor()
)

# Training and testing DataLoader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Set device
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FullBinaryDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.BinaryLinearLayer1 = BinaryLinear(28 * 28, 128)
        self.BinaryLinearLayer2 = BinaryLinear(128, 10)
        self.BinaryReLU = BinaryReLU()
        self.stack = nn.Sequential(
            self.BinaryLinearLayer1,
            self.BinaryReLU,
            self.BinaryLinearLayer2,
        )

    def set_kk(self, kk_new):
        self.BinaryLinearLayer1.set_kk(kk_new)
        self.BinaryLinearLayer2.set_kk(kk_new)
        self.BinaryReLU.set_kk(kk_new)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


model = FullBinaryDense().to(device)  # Move model to the chosen device


def TrainLoop(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    model.train()  # Set model in training mode
    kk_new = torch.Tensor([kk_list[t]]).to(device)
    model.set_kk(kk_new)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):  # Correct 8: Typo corrected in variable name
        print(f"Epoch {t + 1}/{epochs}, k = {kk_list[t]}\n-------------------------------")
        TrainLoop(train_dataloader, model, loss_fn, optimizer, t)
        TestLoop(test_dataloader, model, loss_fn, t)
    print("Done!")

    torch.save(model.state_dict(), 'MINIST_full_binary.pth')

    plt.plot(torch.arange(1, epochs + 1), accuracy, 'o-')
    # 为每个点添加标签
    for i, label in enumerate(kk_list):
        plt.text(i + 0.9, accuracy[i] + 0.001, label, fontsize=7)
    plt.show()
