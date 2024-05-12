import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
# class Simpson_loader(Dataset):
#     def __init__():
#         super(Simpson_loader,self).__init()
#     def __len__():
        
        
        
        
if __name__=='__main__':
    dataset=ImageFolder('data/simpsons_dataset',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
    dataloader=DataLoader(dataset,batch_size=8,shuffle=True)
    for image,label in dataloader:
        print(image)
        print(label)