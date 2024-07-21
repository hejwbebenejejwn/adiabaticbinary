import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from modules.cutout import Cutout
from transform_lmdb import ImageFolderLMDB
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

def read_dataset(batch_size=128,valid_size=0.2,num_workers=10,subset=True):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    
    transform=transforms.Resize((224,224))


    if subset:
        pic_path='D:/usr14/project/Binary/imagenet1/subset'
        dataset=datasets.ImageFolder(pic_path,transform)
        valset=datasets.ImageFolder(pic_path,transform)

        num_samples=len(dataset)
        indices=list(range(num_samples))
        np.random.shuffle(indices)
        split=int(np.floor(valid_size*num_samples))
        train_idx, val_idx, test_idx=indices[2*split:],indices[split:2*split],indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(val_idx)
        test_sampler=SubsetRandomSampler(test_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, 
            sampler=val_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, 
            sampler=test_sampler, num_workers=num_workers)
    else:
        train_path="/data/home/wwang/projs/binary/data/imagenet/train.lmdb"
        test_path="/data/home/wwang/projs/binary/data/imagenet/val.lmdb"
        trainset = ImageFolderLMDB(train_path, transform)
        testset=ImageFolderLMDB(test_path,transform)

        num_samples=len(trainset)
        indices=list(range(num_samples))
        np.random.shuffle(indices)
        split=int(np.floor(valid_size*num_samples))
        train_idx, val_idx=indices[split:],indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
            sampler=val_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
            num_workers=num_workers)

    return train_loader,valid_loader,test_loader




if __name__ == "__main__":
    transform = Resize((224, 224))
    path = "/data/home/wwang/projs/binary/data/imagenet/train.lmdb"
    dataset = ImageFolderLMDB(path, transform)
    data_loader = DataLoader(dataset, batch_size=256, num_workers=10)
    for img, label in data_loader:
        # pass
        print(img.shape)
        print(label.shape)