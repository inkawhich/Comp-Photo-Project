# NAI

from __future__ import print_function 
from __future__ import division
import numpy as np
import os
import copy
import random

######################################################################
# Helper Functions 
######################################################################
# Parse a log file into a list of lists. The returned list is formatted as:
#  [ [ cls, sync, [logit array]],
#    [ cls, sync, [logit array]],
#    ...
#  ]
def parse_log_file(f):
	lines = []
	print("Parsing: ",f)
	logfile = open(f,"r")
	for line in logfile:
		# Parse line into list of floats	
		parts = [float(x) for x in line.rstrip().split(",")]
		# Extract logits into their own list
		logits = [x for x in parts[2:]]
		# Add entry into lines
		lines.append([int(parts[0]), parts[1], logits])
	logfile.close()
	return lines

def test_ensemble(model_predictions, model_weights, num_classes=7):
	
	assert(len(model_predictions) == len(model_weights))
	
	# Initialize Stat Keepers
	accuracy_cnt = 0.
	total = 0
	cmat = np.zeros((num_classes,num_classes))
	
	# For each test data point
	print("Num Test Data: ",len(model_predictions[0]))
	for i in range(len(model_predictions[0])):
		preds = []
		gt = model_predictions[0][i][0]
		print("GT Class: ",gt)
		print(i)
	
		# Get the prediction from each model for this datapoint
		for j in range(len(model_predictions)):
			# Make sure models are synced up and the test data is the same
			assert(model_predictions[j][i][0] == gt)	
			# Grab the logit array for this model	
			preds.append(model_predictions[j][i][2])

		# Here, preds contains the logit array for each model in the ensemble
		#  for this datapoint. Now we just have to combine them to get a single prediction
		print(preds)
				
		############################################
		# TODO: Combine Preds in intelligent way
		combined = np.sum(np.array(preds),axis=0)
		print(combined)
		############################################
	
		# Get prediction from combined array		
		p = np.argmax(combined)
	
		# Update stats
		cmat[gt,p] += 1
		if gt == p:
			accuracy_cnt += 1.
		total += 1
	
	# Return the accuracy as a percentage and the confusion matrix
	return accuracy_cnt/total, cmat



######################################################################
# Inputs 
######################################################################

# Seed for random number generator. This is what defines split 01 and 02
SEED=12345  # Split-01
#SEED=56789  # Split-02
#SEED=63751  # Split-03

# Choose the models that will make up the ensemble
log_root_dir = "model_predictions/val/seed{}".format(SEED)
model_files = [ 
	#log_root_dir+"/resnet_dark_100epoch_seed{}_testlogits.txt".format(SEED),
	log_root_dir+"/resnet_lime_100epoch_seed{}_testlogits.txt".format(SEED),
	log_root_dir+"/resnet_multiscaleRetinex_100epoch_seed{}_testlogits.txt".format(SEED),
	#log_root_dir+"/resnet_BIMEF_100epoch_seed{}_testlogits.txt".format(SEED),
	#log_root_dir+"/resnet_ying_100epoch_seed{}_testlogits.txt".format(SEED),
	]

# Weights for the ensembled models
weights = [.5, .5]

assert(len(model_files) == len(weights))
assert(np.sum(weights) == 1.)

######################################################################
# Ensemble Models
######################################################################

# Parse log files and get predictions
model_preds = []
for log in model_files:
	model_preds.append(parse_log_file(log))

# TODO: do weight sweeps to get near optimal weighting

# Test the ensemble
acc, cmat = test_ensemble(model_preds,weights)

#### Analyze cmat for per class stats
print(acc)
print(cmat)
class_gt_totals = np.sum(cmat,axis=1)
class_pred_totals = np.sum(cmat,axis=0)
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


