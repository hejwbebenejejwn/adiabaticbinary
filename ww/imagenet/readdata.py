import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='D:/usr14/project/Binary/imagenet1/subset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4), 
        # transforms.RandomHorizontalFlip(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])


    dataset=datasets.ImageFolder(pic_path,transform_train)

    num_samples=len(dataset)
    indices=list(range(num_samples))
    np.random.shuffle(indices)
    split=int(np.floor(.2*num_samples))
    train_idx, val_idx, test_idx=indices[2*split:],indices[split:2*split],indices[:split]
    train_sampler=SubsetRandomSampler(train_idx)
    val_sampler=SubsetRandomSampler(val_idx)
    test_sampler=SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        sampler=test_sampler, num_workers=num_workers)


    return train_loader,valid_loader,test_loader