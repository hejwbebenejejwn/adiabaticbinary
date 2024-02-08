import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from Full_Binary_MINIST_Train import FullBinaryDense
from Full_Precision_MINIST_Train import DenseNet
from BinaryFunctions import BinaryHeaviside

from time import time
from time import sleep

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


def ModelUsage(data, model2, model3):
    cols = 3
    rows = 3
    figure2, axarr2 = plt.subplots(rows, cols, figsize=(12, 12))
    figure3, axarr3 = plt.subplots(rows, cols, figsize=(12, 12))
    total_time2 = 0
    total_time3 = 0
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]

        # use model to predict the picture
        img = img.to(device)
        with torch.no_grad():
            t1 = time()
            pred2 = model2(img)
            sleep(0.01)
            t2 = time()
            total_time2 += t2 - t1

            t1 = time()
            pred3 = model3(img)
            sleep(0.01)
            t2 = time()
            total_time3 += t2 - t1
        result2 = torch.argmax(pred2)
        result3 = torch.argmax(pred3)

        # 计算当前子图的索引
        idx = i - 1  # 转换为0-based索引
        ax2 = axarr2[idx // cols, idx % cols]
        ax2.set_title(f'Predict Result: {result2.item()} \nReal Value: {label}')
        ax2.axis("off")
        ax2.imshow(img.squeeze().cpu(), cmap="gray")  # 确保img在CPU上

        ax3 = axarr3[idx // cols, idx % cols]
        ax3.set_title(f'Predict Result: {result3.item()} \nReal Value: {label}')
        ax3.axis("off")
        ax3.imshow(img.squeeze().cpu(), cmap="gray")
    # 单独显示每个figure
    print(f'Binary Time: {total_time2 - 0.09:.5f} s')
    print(f'Full Precision Time: {total_time3 - 0.09:.5f} s')
    figure2.suptitle(f'Binary Time: {total_time2 - 0.09:.5f} s', fontsize=16)
    figure3.suptitle(f'Full Precision Time: {total_time3 - 0.09:.5f} s', fontsize=16)
    plt.figure(figure2.number)
    plt.show()

    plt.figure(figure3.number)
    plt.show()


if __name__ == "__main__":
    model1 = FullBinaryDense().to(device)  # Instantiate the model
    model1.load_state_dict(torch.load('MINIST_binary_weights.pth'))  # Load the state dict
    model1.eval()
    model1 = model1.to(device)
    # 存储权重和偏置
    parameter_list = []
    for name, param in model1.named_parameters():
        if param.requires_grad:
            parameter_list.append(param.data)
    # 二值化权重
    for i in [0, 2]:
        parameter_list[i] = torch.sign(parameter_list[i] * model1.BinaryLinearLayer1.kk)
    # 尝试使用二值化权重激活全连接神经网络
    model2 = DenseNet().to(device)  # Instantiate the model
    model2.stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            BinaryHeaviside(),
            torch.nn.Linear(128, 10)
        )
    new_state_dict = model2.state_dict()
    # 遍历原模型的权重，只复制新模型中存在的权重
    j = 0
    for name, param in model2.named_parameters():
        if name in new_state_dict:
            new_state_dict[name].copy_(parameter_list[j])
            j += 1
    model2.eval()
    print(model2)
    for name, param in model2.named_parameters():
        if param.requires_grad:
            print(name)
            print(param)

    # 全精度全连接神经网络
    model3 = DenseNet().to(device)  # Instantiate the model
    model3.load_state_dict(torch.load('model_weights.pth'))  # Load the state dict
    model3.eval()

    ModelUsage(test_data, model2, model3)