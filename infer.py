from model import ResNet
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
model=ResNet(224)
#load model
model_weights='weights/Resnet_epoch_59_loss:2942.Resnet_epoch_59_loss:2942.817626953125'
test_path='data/kaggle_simpson_testset/kaggle_simpson_testset'
train_path='data/simpsons_dataset'
device='cuda' if torch.cuda.is_available() else 'cpu'
batch=32

# data=ImageFolder(test_path,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
# # data=ImageFolder(train_path)
# dataloader=DataLoader(data,batch_size=batch,shuffle=False)

model.eval()

for img_path in sorted(os.listdir(test_path)):
    # print(type(img_path[0]))
    img=Image.open(img_path)
    img=
    
    
    
    