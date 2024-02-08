import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Some training constants
learning_rate = 5e-1
batch_size = 60
epochs = 20

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


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


model = DenseNet().to(device)  # Move model to the chosen device


def TrainLoop(dataloader, model, loss_fn, optimizer):
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


def TestLoop(dataloader, model, loss_fn):
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


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):  # Correct 8: Typo corrected in variable name
        print(f"Epoch {t + 1}\n-------------------------------")
        TrainLoop(train_dataloader, model, loss_fn, optimizer)
        TestLoop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights.pth')
