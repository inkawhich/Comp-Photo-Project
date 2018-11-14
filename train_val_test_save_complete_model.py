# NAI

# Comp Photo Trainer Script 
# This file contains the train and test code for handling a COMBINATION of all datasets
# Models are pretrained from ImageNet

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from helper_functions import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Inputs
######################################################################

data_root = "/root/hostCurUser/Comp-Photo-Data"
dark_dir = data_root+"/Dark-Image-Data"
lime_dir = data_root+"/Dark-Image-Data_lime"
msr_dir = data_root+"/Dark-Image-Data_multiscaleRetinex"
bimef_dir = data_root+"/Dark-Image-Data_BIMEF"
ying_dir = data_root+"/Dark-Image-Data_Ying_2017_ICCV"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 100

# Number of epochs to train for 
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

# Seed for random number generator. This is what defines split 01 and 02
#SEED=12345  # Split-01
#SEED=56789  # Split-02
SEED=63751  # Split-03

random.seed(SEED)
torch.manual_seed(SEED)

# Should we save the final model
save_flag = True
checkpoint_name = "saved_models/complete_{}_{}epoch_seed{}_model.pth.tar".format(model_name,num_epochs,SEED)


######################################################################
# Load Data
######################################################################

# Data augmentation and normalization for training
# Just normalization for validation
input_size = 299 if model_name == "inception" else 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Initialize Dataset
# Here, the dataset can be thought of as a list of (data,lbl) tuples which can
#   be indexed like any other list. So, we create a train/test split by selecting
#   a set of random indexes for training samples and the other indexes are for 
#   the test data.
print("Initializing Datasets...")

dark_dataset = datasets.ImageFolder(dark_dir, transform=data_transforms['val'])
lime_dataset = datasets.ImageFolder(lime_dir, transform=data_transforms['val'])
msr_dataset = datasets.ImageFolder(msr_dir, transform=data_transforms['val'])
bimef_dataset = datasets.ImageFolder(bimef_dir, transform=data_transforms['val'])
ying_dataset = datasets.ImageFolder(ying_dir, transform=data_transforms['val'])

num_classes = len(dark_dataset.classes)
print("Length of Dark Dataset: ",len(dark_dataset))
print("Classes In Dataset: ", dark_dataset.classes)

# Assemble Training,Validation, and Testing Index Lists
train_percentage = .8
val_percentage = .1
test_percentage = .1
train_indexes = random.sample( range(len(dark_dataset)) , int(len(dark_dataset)*train_percentage) )
nottrain_indexes = [x for x in range(len(dark_dataset)) if x not in train_indexes]
val_indexes = random.sample( nottrain_indexes , int(len(dark_dataset)*val_percentage) )
test_indexes = [x for x in nottrain_indexes if x not in val_indexes]

assert((train_percentage+val_percentage+test_percentage)==1.)
assert(set(train_indexes).isdisjoint(test_indexes))
assert(set(train_indexes).isdisjoint(val_indexes))
assert(set(test_indexes).isdisjoint(val_indexes))
assert((len(test_indexes)+len(train_indexes)+len(val_indexes)) == len(dark_dataset))

print("# training: {}".format(len(train_indexes)))
print("# val: {}".format(len(val_indexes)))
print("# test: {}".format(len(test_indexes)))
print("total: {}".format(len(train_indexes)+len(val_indexes)+len(test_indexes)))
print("First 10 test indexes:\n",test_indexes[:10])
print("Last 10 test indexes:\n",test_indexes[-10:])
verify_indexes_with_seed(SEED,test_indexes[:5],test_indexes[-5:])


# Create big train set by concatenating the train parts of each dataset
dark_train = torch.utils.data.Subset(dark_dataset, train_indexes)
dark_val = torch.utils.data.Subset(dark_dataset, val_indexes)
dark_test = torch.utils.data.Subset(dark_dataset, test_indexes)

lime_train = torch.utils.data.Subset(lime_dataset, train_indexes)
lime_val = torch.utils.data.Subset(lime_dataset, val_indexes)
lime_test = torch.utils.data.Subset(lime_dataset, test_indexes)

msr_train = torch.utils.data.Subset(msr_dataset, train_indexes)
msr_val = torch.utils.data.Subset(msr_dataset, val_indexes)
msr_test = torch.utils.data.Subset(msr_dataset, test_indexes)

bimef_train = torch.utils.data.Subset(bimef_dataset, train_indexes)
bimef_val = torch.utils.data.Subset(bimef_dataset, val_indexes)
bimef_test = torch.utils.data.Subset(bimef_dataset, test_indexes)

ying_train = torch.utils.data.Subset(ying_dataset, train_indexes)
ying_val = torch.utils.data.Subset(ying_dataset, val_indexes)
ying_test = torch.utils.data.Subset(ying_dataset, test_indexes)

# Create full sets as concatenations of each smaller set
trainset = torch.utils.data.ConcatDataset([dark_train,lime_train,msr_train,bimef_train,ying_train])
trainset.transform = data_transforms['train']
valset = torch.utils.data.ConcatDataset([dark_val,lime_val,msr_val,bimef_val,ying_val])
valset.transform = data_transforms['val']
testset = torch.utils.data.ConcatDataset([dark_test,lime_test,msr_test,bimef_test,ying_test])
testset.transform = data_transforms['val']

# Create dataloaders for training and test data
loaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8),
           'val': torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8),
           'test': torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)}


######################################################################
# Create Model, Optimizer, Setup Loss, and run training step
######################################################################

# Initialize the model for this run
model_ft,_ = initialize_model(model_name, num_classes, feature_extract, pt=True)
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
#def train_model(model, device, criterion, optimizer, dataloaders, num_epochs=25, is_inception=False):
model_ft, hist = train_model(model_ft, device, criterion, optimizer_ft, loaders, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Test Model
print("Testing Final Model...")
test_accuracy = test_model(model_ft,device,loaders['test'])
print("Final Test Accuracy: {}".format(test_accuracy))

# Save Model
if save_flag:
    state = {'epoch':num_epochs,
             'arch':model_name,
             'finetune': not feature_extract,
             'test_acc':test_accuracy,
             'state_dict':model_ft.state_dict()
             }
    torch.save(state,checkpoint_name)


"""
# Plot the training curves of validation accuracy vs. number 
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = [h.cpu().numpy() for h in hist]
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
"""


