import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data

dir = '/media/ubuntu/0f083fd5-b631-4342-9812-7e262eaff979/gaoyi/datasets/'

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
                                 ]),
    'val': transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
                               ]),
    }

train_dataset = datasets.ImageFolder(root=dir+'/train', transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root=dir+'/val', transform=data_transform['val'])

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)