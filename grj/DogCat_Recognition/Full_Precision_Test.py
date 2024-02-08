import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from Data_Read import DogCatData

from Full_Precision import FullPrecisionCNN

from time import time

# Setting KMP_DUPLICATE_LIB_OK environment variable (Use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 30
label_dict = {"Cat": 0, "Dog": 1}

# import training data and test data
training_dataset = DogCatData(".\\Training_Data\\")
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DogCatData(".\\Test_Data\\")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
        img = torch.reshape(img, [1,img.shape[0],img.shape[1],img.shape[2]])
        with torch.no_grad():
            pred = model(img)
        result = list(label_dict.keys())[torch.argmax(pred)]
        label = list(label_dict.keys())[label]

        figure.add_subplot(rows, cols, i)
        plt.title(f'Predict Result: {result} \nReal Value: {label}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    model = FullPrecisionCNN().to(device)  # Instantiate the model
    model.load_state_dict(torch.load('full_precision_weights.pth'))  # Load the state dict
    model.eval()
    model = model.to(device)
    t1 = time()
    ModelUsage(test_dataset, model)
    t2 = time()
    print(f'Model Usage Time: {t2-t1:.5f} s')