import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.functional as F

##dir = '/home/ubuntu/data/gaoyi/datasets/'
dir = 'G:\mywork\Datasets/'
num_workers = 0
batch_size = 8
data_transform = {
    'train': transforms.Compose([transforms.Resize(256, antialias=True),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                 # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                 ]),
    'val': transforms.Compose([transforms.Resize(256, antialias=True),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                               ]),
    }

train_dataset = torchvision.datasets.CIFAR10(root=dir, train=True, download=False, transform=data_transform['train'])
val_dataset = torchvision.datasets.CIFAR10(root=dir, train=False, download=False, transform=data_transform['val'])

batch_size = batch_size

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
