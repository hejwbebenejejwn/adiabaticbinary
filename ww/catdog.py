import torch

# from Conv_DogCat import CatDog
from modules.Conv_DogCat import CatDog

# from trainer import Trainer
from modules.trainer import Trainer
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def catdog_datagen(path):
    """dog:0 cat:1"""
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=path, transform=transform)

    cat_class_idx = dataset.class_to_idx["cats"]
    dog_class_idx = dataset.class_to_idx["dogs"]

    if cat_class_idx < dog_class_idx:
        dataset.targets = [
            1 if label == cat_class_idx else 0 for label in dataset.targets
        ]
        dataset.class_to_idx["dogs"], dataset.class_to_idx["cats"] = (
            dataset.class_to_idx["cats"],
            dataset.class_to_idx["dogs"],
        )

    return dataset


traval = catdog_datagen("C:/Projects/Binary/wwdata/dogs_vs_cats/train")
train_size = int(0.75 * len(traval))
val_size = len(traval) - train_size
train_dataset, val_dataset = random_split(traval, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

testset = catdog_datagen("C:/Projects/Binary/wwdata/dogs_vs_cats/test")
test_loader = DataLoader(testset, batch_size=32, shuffle=False)


bw = ba = True
print(bw, ba)
savepath = "C:/Projects/Binary/wwdata/dogcat/bestbin.pth"
if bw and not ba:
    mode = "w"
elif ba and not bw:
    mode = "a"
else:
    mode = "b"
print("dogcat, mode: " + mode)
model = CatDog(bw, ba)
# model.load_state_dict(torch.load(savepath))
optz = torch.optim.Adam(model.parameters())
lossfunc = CrossEntropyLoss()
trr = Trainer(1000, mode, model, optz, lossfunc, 0.9)


if mode == "w":
    bw = 1
    ba = 0
    initLR = 0.1
    optz = torch.optim.Aedam(model.parameters(), lr=initLR)

elif mode == "a":
    bw = 0
    ba = 1
    initLR = 0.01
    optz = torch.optim.Adam(model.parameters(), lr=initLR)
    trr.lr_power = 0.3
elif mode == "b":
    bw = ba = 1
    initLR = 0.01
    optz = torch.optim.Adam(model.parameters(), lr=initLR)
    trr.lr_power = 0.3


trr.train(trainloader=train_loader,valloader=val_loader,max_epochs=10,lr=initLR,testloader=test_loader,path=savepath)