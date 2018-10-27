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

# Load trained model
saved_model_path = "saved_models/Dark-Image-Data_lime_resnet_100epoch_model.pth.tar"

# Seed for random number generator. This is what defines split 01 and 02
SEED=12345  # Split-01
#SEED=56789  # Split-02
random.seed(SEED)
torch.manual_seed(SEED)


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

# Get proper image dataset for this trained model
data_root = "/root/hostCurUser/Comp-Photo-Data"
data_dir = data_root+"/Dark-Image-Data"
if "BIMEF" in saved_model_path:
	data_dir = data_dir+"_BIMEF" 
elif "Ying" in saved_model_path:
	data_dir = data_dir+"_Ying_2017_ICCV" 
elif "lime" in saved_model_path:
	data_dir = data_dir+"_lime" 
elif "multiscaleRetinex" in saved_model_path:
	data_dir = data_dir+"_multiscaleRetinex"
print("Model Path: ",saved_model_path) 
print("Data Dir: ",data_dir) 

######################################################################
# Load Data
######################################################################

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

# Seed=12345 ==> [38, 73, 77, 86, 98, 114, 116, 125, 140, 143]
print("First 10 test indexes:\n",test_indexes[:10])
# Seed=12345 ==> [4412, 4419, 4420, 4431, 4433, 4441, 4442, 4461, 4462, 4463]
print("Last 10 test indexes:\n",test_indexes[-10:])

# Create dataloaders for training and test data
#valset = torch.utils.data.Subset(full_dataset, val_indexes)
testset = torch.utils.data.Subset(full_dataset, test_indexes)
#valset.transform = data_transforms['val']
testset.transform = test_transform
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

# Run test to make sure model loaded properly
test_accuracy = test_model(model,device,test_loader)
print("Test Acc: ",test_accuracy)
assert(test_accuracy.item() == checkpoint['test_acc'])


######################################################################
# Analyze
######################################################################

# Initialize Stat Keepers
accuracy_cnt = 0.
total = 0
confusion_matrix = np.zeros((7,7))
class_wise_recall_cnt = np.zeros((7))
class_wise_recall_total = np.zeros((7))

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
		confusion_matrix[gt,pred] += 1
		class_wise_recall_total[gt] += 1.
		if gt == pred:
			class_wise_recall_cnt[gt] += 1.
			accuracy_cnt += 1.
		total += 1

tpr_arr = class_wise_recall_cnt/class_wise_recall_total
acc = accuracy_cnt/total

print("Confusion Matrix\n",confusion_matrix)
print("TPR Array\n",tpr_arr)
print("Accuracy: {}".format(acc))

assert(acc == checkpoint['test_acc'])





