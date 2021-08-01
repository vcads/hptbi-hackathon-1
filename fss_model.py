################################################################################
#
################################################################################
# FSS Model
#
# Args:
#
# Return:
#
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def fss_model(data):
    ############################################################################
    # User code starts here
    
    rtn = []
	dataset = data.to_numpy()
	fss_datasets = [dataset[:, 0:7], dataset[:, 7:14], dataset[:, 14:29], dataset[:, 29:50], dataset[:, 50:71], dataset[:, 71:88]]
	fss_scores = dataset[:, 88:94]

	for i in range(len(fss_datasets)):
		target = fss_scores[:, i]
		target_data = []
		for j in range(len(target)):
			target_data.append(target[j])

		x_train, x_test, y_train, y_test = train_test_split(fss_datasets[i], target_data, test_size = 0.3, random_state = 100)
		classifier = RandomForestClassifier(max_depth = 6, min_samples_leaf = 1, min_samples_split = 3, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 150)
		classifier.fit(x_train, y_train)
		rtn.append(classifier)
        
    # User code ends here
    ############################################################################
    return rtn


################################################################################
# Predict Hackathon Mortality Model
#
# Args:
#
# Return:
#   predicted FSS values, integers values inclusively between 6 and 30
#
def predict_fss(model, newdata):
    ############################################################################
    # user defined code starts here
    dataset = newdata.to_numpy()
	fss_datasets = [dataset[:, 0:7], dataset[:, 7:14], dataset[:, 14:29], dataset[:, 29:50], dataset[:, 50:71], dataset[:, 71:88]]
	fss_scores = dataset[:, 88:94]
	predictions = []
	for i in range(len(models)):
		y_pred = models[i].predict(fss_datasets[i])
		predictions.append(y_pred)

	fss_vals = []
	for i in range(len(predictions[0])):
		fss_sum = 0
		for j in range(len(predictions)):
			fss_sum += predictions[j][i]
		fss_vals.append(fss_sum)

	return fss_vals

################################################################################
#                                 End of File
################################################################################
