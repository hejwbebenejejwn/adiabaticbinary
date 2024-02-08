import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from Full_Precision_MINIST_Train import DenseNet

from time import time

# Setting KMP_DUPLICATE_LIB_OK environment variable (Use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# Set device
device = (
    "cpu"
)


def ModelUsage(data, model):
    cols = 3
    rows = 3
    figure = plt.figure(figsize=(12, 12))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]

        # use model to predict the picture
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)
        result = torch.argmax(pred)

        figure.add_subplot(rows, cols, i)
        plt.title(f'Predict Result: {result} \nReal Value: {label}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    model = DenseNet().to(device)  # Instantiate the model
    model.load_state_dict(torch.load('model_weights.pth'))  # Load the state dict
    model.eval()
    model = model.to(device)
    t1 = time()
    ModelUsage(test_data, model)
    t2 = time()
    print(f'Model Usage Time: {t2-t1:.5f} s')