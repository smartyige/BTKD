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
                                 # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                 ]),
    'val': transforms.Compose([transforms.Resize(256, antialias=True),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                               ]),
    }

train_dataset = torchvision.datasets.CIFAR100(root=dir, train=True, download=False, transform=data_transform['train'])
val_dataset = torchvision.datasets.CIFAR100(root=dir, train=False, download=False, transform=data_transform['val'])

batch_size = batch_size

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
