import torch
import torchvision
import random
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from simplemodel import *
from Unet import  *

# Detect if we have a GPU available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("Device: ",device)
##############################################################3
# Inputs
##############################################################3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
HEIGHT=WIDTH=224
DATA_ROOT = "."

SEED=631
random.seed(SEED)
torch.manual_seed(SEED)

##############################################################3
# Format datasets
##############################################################3
#transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop((224,224)), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop((224,224)),transforms.ToTensor()])

# Look at dark dataset and get indexes of training and validation images
print("Initializing Datasets...")
dark_dataset = ImageFolder(DATA_ROOT+'/Dark-Image-Data/', transform=transform)
num_classes = len(dark_dataset.classes)
print("Length of Full Dataset: ",len(dark_dataset))
print("Classes In Dataset: ", dark_dataset.classes)
train_percentage = .9
train_indexes = random.sample( range(len(dark_dataset)) , int(len(dark_dataset)*train_percentage) )
val_indexes = [x for x in range(len(dark_dataset)) if x not in train_indexes]
assert(set(train_indexes).isdisjoint(val_indexes))
print("# training: {}".format(len(train_indexes)))
print("# val: {}".format(len(val_indexes)))
print("total: {}".format(len(train_indexes)+len(val_indexes)))
print("First 10 test indexes:\n",val_indexes[:10])
print("Last 10 test indexes:\n",val_indexes[-10:])

# Create datasets for each algorithm
lime_dataset = ImageFolder(DATA_ROOT+'/Dark-Image-Data_lime/',transform=transform)
msr_dataset = ImageFolder(DATA_ROOT+'/Dark-Image-Data_MSR/',transform=transform)
bimef_dataset = ImageFolder(DATA_ROOT+'/Dark-Image-Data_BIMEF/',transform=transform)
#ying_dataset = ImageFolder(DATA_ROOT+'/Dark-Image-Data_Ying_2017_ICCV/',transform=transform)

# Create training subsets
dark_train = torch.utils.data.Subset(dark_dataset, train_indexes)
lime_train = torch.utils.data.Subset(lime_dataset, train_indexes)
msr_train = torch.utils.data.Subset(msr_dataset, train_indexes)
bimef_train = torch.utils.data.Subset(bimef_dataset, train_indexes)
#ying_train = torch.utils.data.Subset(ying_dataset, train_indexes)

# Create a validation split of the dark data to look at
dark_val = torch.utils.data.Subset(dark_dataset, val_indexes)

# Create dataloaders
dark_loader=DataLoader(dataset=dark_train,batch_size=BATCH_SIZE,num_workers=4,shuffle=False,drop_last=False)
lime_loader=DataLoader(dataset=lime_train,batch_size=BATCH_SIZE,num_workers=4,shuffle=False,drop_last=False)
msr_loader=DataLoader(dataset=msr_train,batch_size=BATCH_SIZE,num_workers=4,shuffle=False,drop_last=False)
bimef_loader=DataLoader(dataset=bimef_train,batch_size=BATCH_SIZE,num_workers=4,shuffle=False,drop_last=False)
#ying_loader=DataLoader(dataset=ying_train,batch_size=BATCH_SIZE,num_workers=4,shuffle=False,drop_last=False)


##############################################################3
# Train AE
##############################################################3
model = UNet(n_channels=3, n_classes=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, weight_decay=1e-5)


for epoch in range(NUM_EPOCHS):
    
    iter_cnt = 0 
    
    for dark_data,lime_data,msr_data in zip(dark_loader,lime_loader,msr_loader):
    # for dark_data, bimef_data in zip(dark_loader, bimef_loader):
    #     iter_cnt += 1
    #
        dark_img, _ = dark_data
        lime_img, _ = lime_data
        msr_img, _ = msr_data
        # bimef_img, _ = bimef_data
        #ying_img, _ = ying_data
        #light_img = bimef_img

        # Choose what GT to penalize against
        gt_data = random.randint(0,1)
        if gt_data == 0:
            light_img = lime_img
        elif gt_data == 1:
            light_img = msr_img
        # elif gt_data == 2:
        #     light_img = bimef_img
        # elif gt_data == 3:
        #     light_img = ying_img
        
        decoded = model(dark_img.to(device))
        loss = criterion(decoded,light_img.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("[Epoch: {} / {}][Iter: {} / {}] Loss = {}".format(epoch,NUM_EPOCHS,iter_cnt,int(len(dark_train)/BATCH_SIZE),loss.data[0]))

torch.save(model.state_dict(),'./torch_model/autoencoder_unet_full.pth')
