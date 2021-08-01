################################################################################
# Prepare Mortality Data
#
# Define data procesing steps to apply to the data set used to train and test
# models for predicting mortality.
#
# Args:
#   training  (logicial) if the data set to read in is the training or testing
#             data set.
#
# Return:
#   A pandas data.frame with the defnined primary outcome and any user specific
#   elements needed for training and testing their model.
#

import pandas as pd
import numpy as np
import pathlib
import re
from sklearn.preprocessing import LabelEncoder

def prepare_mortality_data(training = True):
    training_data = pathlib.Path("./csvs/training.csv")
    testing_data  = pathlib.Path("./csvs/testing.csv")

    if not training and testing_data.exists():
        hackathon_mortality_data = pandas.read_csv(testing_data)
    else:
        hackathon_mortality_data = pandas.read_csv(training_data)

    # Define the primary outcome -- do not edit this.  If you need the outocme
    # in a different format, e.g., integer or logical, create an additional
    # data.frame element in the user defined code section below.
    hackathon_mortality_data["mortality"] = hackathon_mortality_data["hospdisposition"] == "Mortality"
    hackathon_mortality_data["mortality"] = hackathon_mortality_data["mortality"].astype(int)

    # Omit some elements - FSS is omitted from this data set.  FSS could not be
    # assessed for patients who died.  To reduce confusion FSS related elements
    # are omitted as missing values for FSS are be highly correlated with
    # mortality.
    #for c in hackathon_mortality_data.filter(regex = "fss").columns:
    #    hackathon_mortality_data = hackathon_mortality_data.drop(columns = c)
    hackathon_mortality_data = hackathon_mortality_data.filter(regex = "^(?!.*fss.*)")

    ##############################################################################
    # User Defined Code starts here
    hackathon_mortality_data['cardiacarrest_sum'] = hackathon_mortality_data['cardiacarrestyn'] + hackathon_mortality_data['cardiacarrestprehosp'] + hackathon_mortality_data['cardiacarrested'] + hackathon_mortality_data['cardiacarrestor'] + hackathon_mortality_data['cardiacarresticu'] + hackathon_mortality_data['cardiacarrestother']

	weird_labels = hackathon_mortality_data.columns # label names
	labels = [] # contains label names in a clean way

	# Adds the column names to labels
	for i in range(len(weird_labels)):
		labels.append(weird_labels[i])

	# Converts hackathon_mortality_data to an easier to work with NumPy array
	hackathon_mortality_data = hackathon_mortality_data.to_numpy()

	num_ppl = len(hackathon_mortality_data)

	# Features that I am going to use
	features = ['age', 'admittoentnut', 'entnutyn', 'hosplos', 'puplrcticu', 'gcsicu', 'cardiacarrestyn', 'cardiacarrest_sum']
		
	# Final feature names I am going to use
	final_features = ['age', 'puplrcticu', 'gcsicu', 'entnutyn', 'admittoentnut', 'hosplos', 'cardiacarrestyn', 'cardiacarrest_sum', 'mortality']

	num_features = len(features) + 1

	# Finds column indexes for each feature
	for i in range(len(features)):
		features[i] = labels.index(features[i])

	mort_col = labels.index('mortality')

	# Orders features in ascending order of indices
	features.sort()

	# Splits the dataset into the necessary values
	dataset = hackathon_mortality_data[:, features]
	target = hackathon_mortality_data[:, mort_col]

	new_target = []
	for i in range(len(target)):
		new_target.append([target[i]])
	dataset = np.append(dataset, new_target, axis = 1)

	# Finds which columns contains strings
	str_cols = set()
	for i in range(num_ppl):
		for j in range(num_features):
			if type(dataset[i][j]) == str:
				str_cols.add(j)

	# Converts the strings to integers via LabelEncoder
	for col in str_cols:
		str_list = [] # contains the strings for each column
		for i in range(num_ppl):
			str_list.append(dataset[i][col])
		label_encoder = LabelEncoder()
		new_vals = label_encoder.fit_transform(str_list) # converts to ints
		for i in range(num_ppl):
			dataset[i][col] = int(new_vals[i]) # replaces the strs with ints

	# Converts the NaNs (missing values) to -1's
	for i in range(num_ppl):
		for j in range(num_features):
			if np.isnan(dataset[i][j]):
				dataset[i][j] = -1
			else:
				dataset[i][j] = int(dataset[i][j])

	# Converts the NumPy array to the required pandas data frame
	hackathon_mortality_data = pd.DataFrame(dataset, columns = final_features)

    # User Defined Code ends here
    ##############################################################################

    return hackathon_mortality_data

################################################################################
#                                 End of File
###############################################################################.
