import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
from model import ResNet
from tqdm import tqdm
import os

epoch=500
batch=39

device='cuda' if torch.cuda.is_available() else 'cpu'
model_path=None
weights_dir='weights'
# print(device)
data_dir='data/simpsons_dataset'
dataset=ImageFolder(data_dir,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
print(f'the class to index {dataset.class_to_idx}')
dataloader=DataLoader(dataset,batch_size=batch,shuffle=True)
# model=ResNet(224).to(device)
model=models.resnet50(weights='DEFAULT').to(device)
model.fc = nn.Linear(2048,42)  # Adjust the final layer to match the number of classes
model = model.to(device)
optimizer=optim.Adam(model.parameters(),0.001) 
epoch_loss=0
running_loss=0

if model_path:
    print(f' model loaded :{model_path}')
    model.load_state_dict(torch.load(model_path,map_location=device))
num_imgs=len(dataset.imgs)
criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.1*(1-len(os.listdir(os.path.join(data_dir,file)))/num_imgs) for file in sorted(os.listdir(data_dir))],device=device))
model.train()
for i in tqdm(range(epoch),unit='epoch'):
    epoch_loss=0
    for count,data in tqdm(enumerate(dataloader),unit='iteration'):
        optimizer.zero_grad()
        images=data[0]
        # print(f'image shape is :{images.shape}')
        label=data[1]
        # print(f'label tensor is {label}')
        # print(f'label shape is :{label.shape}')
        images=images.to(device)
        label=label.to(device)
        output=model(images)
        
        l=criterion(output,label)
      
        epoch_loss+=l.item()
        # if count%50==0:
        
        print(f'\n[ epoch :{i} iteration  :{count} loss  :{l}]')
        l.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f'{name}: {param.grad.abs().mean().item()}')

    running_loss+=epoch_loss
    
    print(f'[epoch loss:{epoch_loss}   running loss:{running_loss/(i+1)}]\n')
    torch.save(model.state_dict(),f'{weights_dir}/Resnet_epoch_{i}_loss:{epoch_loss}')
