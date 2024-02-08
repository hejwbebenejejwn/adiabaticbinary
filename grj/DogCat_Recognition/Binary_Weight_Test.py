import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from Data_Read import DogCatData

from Binary_Weight import BinaryWeightCNN
from Full_Precision import FullPrecisionCNN
from BinaryFunctions import BinaryHeaviside

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
    model1 = BinaryWeightCNN().to(device)  # Instantiate the model
    model1.load_state_dict(torch.load('binary_weight_weights.pth'))  # Load the state dict
    model1.eval()
    model1 = model1.to(device)

    # 存储权重和偏置
    parameter_list = []
    for name, param in model1.named_parameters():
        if param.requires_grad:
            parameter_list.append(param.data)
    # 二值化权重
    for i in [0, 4, 8, 12]:
        parameter_list[i] = torch.sign(parameter_list[i] * model1.dense.kk)

    # 尝试使用二值化权重激活全连接神经网络
    model2 = FullPrecisionCNN().to(device)  # Instantiate the model
    new_state_dict = model2.state_dict()

    # 遍历原模型的权重，只复制新模型中存在的权重
    j = 0
    for name, param in model2.named_parameters():
        if name in new_state_dict:
            new_state_dict[name].copy_(parameter_list[j])
            j += 1
    model2.eval()

    for name, param in model2.named_parameters():
        if param.requires_grad:
            print(name)
            print(param)

    t1 = time()
    ModelUsage(test_dataset, model2)
    t2 = time()
    print(f'Model Usage Time: {t2-t1:.5f} s')