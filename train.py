import torch
# from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import ResNet
batch=12
dataset=ImageFolder('data/simpsons_dataset',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
dataloader=DataLoader(dataset,batch_size=batch,shuffle=True)
model=ResNet(224)
optimizer=optim.adam()
