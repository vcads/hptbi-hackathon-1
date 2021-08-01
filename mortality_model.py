################################################################################
#
################################################################################
# Mortality Model
#
# Args:
#
# Return:
#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # for splitting into training and test set
from sklearn import svm # for modeling

def mortality_model(data):

    ymat = data.loc[:, "mortality"]

    ############################################################################
    # User code starts here
    
    dataset = data.to_numpy()

	target_data = []
	for i in range(300):
		if dataset[i][8] == 1:
			target_data.append(1)
		else:
			target_data.append(0)

	dataset = dataset[:, 0:8]

	x_train, x_test, y_train, y_test = train_test_split(dataset, target_data, test_size = 0.3, random_state = 100)

	# creates a SVM classifier
	rtn = svm.SVC(kernel = 'linear', random_state = 200) # linear kernel (2D)

	# trains the model using the training sets
	rtn.fit(x_train, y_train)
    
    # User code ends here
    ############################################################################
    return rtn

################################################################################
# Predict Hackathon Mortality Model
#
# Args:
#
# Return:
#
def predict_mortality(model, newdata):

    ############################################################################
    # User Defined data preparation code starts here
    
    x_test = newdata.to_numpy()
	x_test = x_test[:, 0:8]

	y_pred = model.predict(x_test)

	predictions = []
	for i in range(len(y_pred)):
		if y_pred[i] == 1:
			predictions.append("Mortality")
		else:
			predictions.append("Alive")

	return predictions

################################################################################
#                                 End of File
################################################################################
