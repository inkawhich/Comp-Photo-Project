# NAI

# Comp Photo Trainer Script 
# This file contains the train and test code for handling a single dataset
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

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
#dset = ""
#dset = "_lime"
#dset = "_multiscaleRetinex"
#dset = "_BIMEF"
#dset = "_Ying_2017_ICCV"
#data_dir = "/root/hostCurUser/Comp-Photo-Data/Dark-Image-Data{}".format(dset)
data_dir = "/root/hostCurUser/Comp-Photo-Data/Light-Image-Data"; dset="LIGHT"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 64

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
checkpoint_name = "saved_models/{}_{}_{}epoch_seed{}_model.pth.tar".format(data_dir.split("/")[-1],model_name,num_epochs,SEED)


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

full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
num_classes = len(full_dataset.classes)
print("Length of Full Dataset: ",len(full_dataset))
print("Classes In Dataset: ", full_dataset.classes)

# Assemble Training,Validation, and Testing Index Lists
train_percentage = .8
val_percentage = .1
test_percentage = .1
train_indexes = random.sample( range(len(full_dataset)) , int(len(full_dataset)*train_percentage) )
nottrain_indexes = [x for x in range(len(full_dataset)) if x not in train_indexes]
val_indexes = random.sample( nottrain_indexes , int(len(full_dataset)*val_percentage) )
test_indexes = [x for x in nottrain_indexes if x not in val_indexes]

assert((train_percentage+val_percentage+test_percentage)==1.)
assert(set(train_indexes).isdisjoint(test_indexes))
assert(set(train_indexes).isdisjoint(val_indexes))
assert(set(test_indexes).isdisjoint(val_indexes))
assert((len(test_indexes)+len(train_indexes)+len(val_indexes)) == len(full_dataset))

print("# training: {}".format(len(train_indexes)))
print("# val: {}".format(len(val_indexes)))
print("# test: {}".format(len(test_indexes)))
print("total: {}".format(len(train_indexes)+len(val_indexes)+len(test_indexes)))

print("First 10 test indexes:\n",test_indexes[:10])
print("Last 10 test indexes:\n",test_indexes[-10:])
#verify_indexes_with_seed(SEED,test_indexes[:5],test_indexes[-5:])

# Create dataloaders for training and test data
trainset = torch.utils.data.Subset(full_dataset, train_indexes)
valset = torch.utils.data.Subset(full_dataset, val_indexes)
testset = torch.utils.data.Subset(full_dataset, test_indexes)
trainset.transform = data_transforms['train']
valset.transform = data_transforms['val']
testset.transform = data_transforms['val']
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
print("Data: {}".format(dset))

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


