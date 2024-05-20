from model import ResNet
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision import models
import torch.nn as nn
import os
import matplotlib.pyplot as plt
# model=ResNet(224)
import os
import matplotlib.pyplot as plt
# model=ResNet(224)
#load model
model_weights='weights/Resnet_epoch_6_loss:24.227733492210973'
test_path='data/kaggle_simpson_testset/kaggle_simpson_testset'
train_path='data/simpsons_dataset'
# device='cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
batch=32
data=Dataset()
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224),antialias=True)])
# data=ImageFolder(train_path,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
# # data=ImageFolder(train_path)
# dataloader=DataLoader(data,batch_size=batch,shuffle=False)
# print(type(data))
model=models.resnet50(weights='DEFAULT').to(device)
model.fc = nn.Linear(2048,42)  # Adjust the final layer to match the number of classes
model = model.to(device)
model.eval()
# store=torch.ones((1,42)).to('cuda')
# prifnt(store)
model.load_state_dict(torch.load(model_weights,map_location=device))
print('model loaded')
for img_path in sorted(os.listdir(test_path)):
    # print(type(img_path[0]))
    img=Image.open(os.path.join(test_path,img_path))
    x=transform(img).to(device)
    x=torch.unsqueeze(x,dim=0)
    # print(x.shape)
    y=model(x)
    # print(f'output {y-store}')
       
    # store=y
    # print(y)
    y=torch.argmax(y)
    print(f'image name : {img_path}')
    print(f'the predicted class is :{y}')

    # img.show()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
    
    