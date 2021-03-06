# NAI

# This script takes a single trained model and analyzes it with a confusion
#  matrix and class wise true positive rates

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

# Seed for random number generator. This is what defines split 01 and 02
#SEED=12345  # Split-01
#SEED=56789  # Split-02
SEED=63751  # Split-03
random.seed(SEED)
torch.manual_seed(SEED)

data_root = "/root/hostCurUser/Comp-Photo-Data"
dark_dir = data_root+"/Dark-Image-Data"
lime_dir = data_root+"/Dark-Image-Data_lime"
msr_dir = data_root+"/Dark-Image-Data_multiscaleRetinex"
bimef_dir = data_root+"/Dark-Image-Data_BIMEF"
ying_dir = data_root+"/Dark-Image-Data_Ying_2017_ICCV"
saved_model_path = "saved_models/complete_resnet_100epoch_seed{}_model.pth.tar".format(SEED)

# Save flag for saving output logits to file
# BE CAREFUL! ERROR PRONE LOG FILE NAME!
save_flag=False
if save_flag == True:
	parts = saved_model_path.split("_")
	log_file = "model_predictions/val/seed{}/{}{}_{}_seed{}_testlogits.txt".format(SEED,parts[3],dset,parts[4],SEED)
	#log_file = "model_predictions/val/seed{}/resnet_dark_100epoch_seed{}_testlogits.txt".format(SEED,SEED)
	log = open(log_file,"w")
	print("Log File: ",log_file)

######################################################################
# Load Model
######################################################################

#### Prepare and Initialize the trained model
print("Initializing Model...")
model = models.resnet50(pretrained=False,num_classes=7)
checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint['state_dict'])
print("Saved Model Accuracy: ",checkpoint['test_acc'])
print("Saved Model Train Epochs: ",checkpoint['epoch'])
model.eval()
model.to(device)

######################################################################
# Load Data
######################################################################

"""
# Data augmentation and normalization for training
# Just normalization for validation
input_size = 224
test_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#### Initialize Dataset
print("Initializing Datasets...")

full_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
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
verify_indexes_with_seed(SEED,test_indexes[:5],test_indexes[-5:])

# Create dataloaders for training and test data
############ BE CAREFUL!!
#testset = torch.utils.data.Subset(full_dataset, val_indexes)
testset = torch.utils.data.Subset(full_dataset, test_indexes)
############

#valset.transform = data_transforms['val']
testset.transform = test_transform
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
"""

input_size =  224
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

batch_size=1
# Create dataloaders for training and test data
loaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1),
           'val': torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1),
           'test': torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)}

############ BE CAREFUL!!
#test_loader = loaders["val"]
test_loader = loaders["test"]
############



# Run test to make sure model loaded properly
test_accuracy = test_model(model,device,test_loader)
print("Test Acc: ",test_accuracy.item())
print("Ckpt Acc: ",checkpoint['test_acc'].item())
#assert(abs(test_accuracy.item() - checkpoint['test_acc'].item()) <= 1e-5)

#exit()

######################################################################
# Analyze
######################################################################

# Initialize Stat Keepers
accuracy_cnt = 0.
total = 0
cmat = np.zeros((7,7))

with torch.no_grad():
 
	for inputs, labels in test_loader:
		# Prepare data
		inputs = inputs.float().to(device)
		labels = labels.long()
		# Forward pass and get prediction
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		gt = labels.item()
		pred = preds.item()
		# Update stats
		cmat[gt,pred] += 1
		if gt == pred:
			accuracy_cnt += 1.
		total += 1

		# If save_flag is true, we write the gt label, the first pixel value of data which may
		#   be used as a sync value, and the list of class logits.
		if save_flag == True:
			logits = outputs.cpu().squeeze().numpy()
			logit_list = ','.join(str(x) for x in logits)
			line = "{},{},{}\n".format(gt,inputs[0,0,0,0],logit_list)
			log.write(line)

acc = accuracy_cnt/total

# Analyze cmat for per class stats
class_gt_totals = np.sum(cmat,axis=1)
class_pred_totals = np.sum(cmat,axis=0)
print(cmat)
print("Class GT totals: ",class_gt_totals)
print("Class Pred totals: ",class_pred_totals)
tpr_arr = np.zeros((len(cmat)))
precision_arr = np.zeros((len(cmat)))
for i in range(len(cmat)):
	tpr_arr[i] = cmat[i,i]/float(class_gt_totals[i])
	precision_arr[i] = cmat[i,i]/float(class_pred_totals[i])
tmp = (1./tpr_arr)+(1./precision_arr)
f1_arr = 2./tmp
print("TPR Arr: ",tpr_arr)
#print("Precision Arr: ",precision_arr)
print("F1 Arr: ",f1_arr)
print("Accuracy: ",acc)

assert(abs(acc - checkpoint['test_acc'].item()) <= 1e-5)

print(saved_model_path)


""" Testing Stat Metrics by hand
# Test Stat Metrics
cmat = np.array([[10,1,6,2],[0,12,1,1],[4,3,9,1],[0,0,6,13]])
class_gt_totals = np.sum(cmat,axis=1)
class_pred_totals = np.sum(cmat,axis=0)
print(cmat)
print("Class GT totals: ",class_gt_totals)
print("Class Pred totals: ",class_pred_totals)
tpr_arr = np.zeros((len(cmat)))
precision_arr = np.zeros((len(cmat)))
for i in range(len(cmat)):
	tpr_arr[i] = cmat[i,i]/float(class_gt_totals[i])
	precision_arr[i] = cmat[i,i]/float(class_pred_totals[i])
tmp = (1./tpr_arr)+(1./precision_arr)
f1_arr = 2./tmp
print("TPR Arr: ",tpr_arr)
print("Precision Arr: ",precision_arr)
print("F1 Arr: ",f1_arr)
exit()
"""

