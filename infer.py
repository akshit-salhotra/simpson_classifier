from model import ResNet
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import os
import matplotlib.pyplot as plt
model=ResNet(224)
#load model
model_weights='weights/Resnet_epoch_59_loss:2942.Resnet_epoch_59_loss:2942.817626953125'
test_path='data/kaggle_simpson_testset/kaggle_simpson_testset'
train_path='data/simpsons_dataset'
device='cuda' if torch.cuda.is_available() else 'cpu'
batch=32
data=Dataset()
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224),antialias=True)])
# data=ImageFolder(train_path,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
# # data=ImageFolder(train_path)
# dataloader=DataLoader(data,batch_size=batch,shuffle=False)
# print(type(data))
model.eval()

for img_path in sorted(os.listdir(test_path)):
#     # print(type(img_path[0]))
    img=Image.open(os.path.join(test_path,img_path))
    x=transform(img)
    x=torch.unsqueeze(x,dim=0)
    # print(x.shape)
    y=model(x)
    y=torch.argmax(y)
    print(f'image name : {img_path}')
    print(f'the predicted class is :{y}')

    # img.show()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
    
    