import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ResNet
from tqdm import tqdm

epoch=500
batch=32
device='cuda' if torch.cuda.is_available() else 'cpu'
model_path="weights/Resnet_epoch_16_loss:2949.54833984375"
weights_dir='weights'
# print(device)
dataset=ImageFolder('data/simpsons_dataset',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
dataloader=DataLoader(dataset,batch_size=batch,shuffle=True)
model=ResNet(224).to(device)
optimizer=optim.Adam(model.parameters(),0.001) 
criterion=nn.CrossEntropyLoss()
epoch_loss=0
running_loss=0

if model_path:
    print(f' model loaded :{model_path}')
    model.load_state_dict(torch.load(model_path,map_location=device))
    
for i in tqdm(range(17,epoch),unit='epoch'):
    model.train()
    epoch_loss=0
    for count,data in tqdm(enumerate(dataloader),unit='iteration'):
        images=data[0]
        label=data[1]
        images=images.to(device)
        label=label.to(device)
        output=model(images)
        l=criterion(output,label)        
        epoch_loss+=l
        # if count%50==0:
        print(f'\n[ epoch :{i} iteration  :{count} loss  :{l/batch}]')
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
    running_loss+=epoch_loss/len(dataset)
    print(f'[epoch loss:{epoch_loss/len(dataset)}   running loss:{running_loss/i}]\n')
    torch.save(model.state_dict(),f'{weights_dir}/Resnet_epoch_{i}_loss:{epoch_loss}')
    