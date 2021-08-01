################################################################################
# Prepare FSS Data
#
# Define data processing steps to apply to the data set used to train and test
# models for predicting FSS.
#
# Args:
#   training  (logical) if the data set to read in is the training or testing
#             data set.
#
# Return:
#   A pandas data.frame with the defined primary outcome and any user-specific
#   elements needed for training and testing their model.
#

import pandas as pd
import numpy as np
import pathlib
import re
from sklearn.preprocessing import LabelEncoder

def prepare_fss_data(training = True):
    training_data = pathlib.Path("./csvs/training.csv")
    testing_data  = pathlib.Path("./csvs/testing.csv")

    if not training and testing_data.exists():
        hackathon_fss_data = pandas.read_csv(testing_data)
    else:
        hackathon_fss_data = pandas.read_csv(training_data)

    # Define the primary outcome -- do not edit this.  If you need the outcome in
    # a different format, e.g., integer or logical, create an additional
    # data.frame element in user defined code section below.
    hackathon_fss_data["fss_total"] = hackathon_fss_data["fssmental"] + hackathon_fss_data["fsssensory"] + hackathon_fss_data["fsscommun"] + hackathon_fss_data["fssmotor"] + hackathon_fss_data["fssfeeding"] + hackathon_fss_data["fssresp"]

    # subset to known FSS values
    hackathon_fss_data = hackathon_fss_data[(hackathon_fss_data["hospdisposition"] != "Mortality")]
    hackathon_fss_data = hackathon_fss_data[(hackathon_fss_data["fss_total"].notnull())]

    ##############################################################################
    # User Defined Code starts here
    # Creates new columns for the sums of the following values
	hackathon_fss_data['gcsett_sum'] = hackathon_fss_data['gcsetted'] + hackathon_fss_data['gcsetticu']
	hackathon_fss_data['gcssed_sum'] = hackathon_fss_data['gcsseded'] + hackathon_fss_data['gcssedicu']
	hackathon_fss_data['gcspar_sum'] = hackathon_fss_data['gcspared'] + hackathon_fss_data['gcsparicu']
	hackathon_fss_data['gcseyeob_sum'] = hackathon_fss_data['gcseyeobed'] + hackathon_fss_data['gcseyeobicu']
	hackathon_fss_data['ct_sum'] = hackathon_fss_data['ctskullfrac'] + hackathon_fss_data['ctce'] + hackathon_fss_data['ctmidlineshift'] + hackathon_fss_data['ctcompress'] + hackathon_fss_data['ctintraparhem'] + hackathon_fss_data['ctsubarchhem'] + hackathon_fss_data['ctintraventhem'] + hackathon_fss_data['ctsubhematoma'] + hackathon_fss_data['ctepihematoma']
	hackathon_fss_data['cardiacarrest_sum'] = hackathon_fss_data['cardiacarrestyn'] + hackathon_fss_data['cardiacarrestprehosp'] + hackathon_fss_data['cardiacarrested'] + hackathon_fss_data['cardiacarrestor'] + hackathon_fss_data['cardiacarresticu'] + hackathon_fss_data['cardiacarrestother']

	weird_labels = hackathon_fss_data.columns # label names
	labels = [] # contains label names in a clean way

	# Adds the column names to labels
	for i in range(len(weird_labels)):
		labels.append(weird_labels[i])

	# Converts hackathon_fss_data to an easier to work with NumPy array
	hackathon_fss_data = hackathon_fss_data.to_numpy()


	fss_mental_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittocathend1', 'admittoext']
	fss_sensory_feats = ['age', 'hosplos', 'admittoicudc1', 'admittocathend3', 'admittocathend2', 'admittoicpend1', 'admittoext']
	fss_commun_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittoext', 'admittocathend1', 'admittocathend3', 'admittocathstart2', 'admittoint', 'admittogast', 'admittoicpstart1', 'icptype1', 'gcsed', 'admittocathstart3']
	fss_motor_feats = ['age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos', 'gcsett_sum', 'gcssed_sum', 'gcspar_sum', 'gcseyeob_sum', 'ct_sum', 'cardiacarrest_sum']
	fss_feeding_feats = ['age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos', 'gcsett_sum', 'gcssed_sum', 'gcspar_sum', 'gcseyeob_sum', 'ct_sum', 'cardiacarrest_sum']
	fss_resp_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoext', 'admittocathend2', 'admittogast', 'admittoicpend1', 'admittocathend1', 'admittoint', 'admittotrach', 'admittocathend3', 'newtrachyn', 'admittocathstart3', 'gcsed', 'admittoicpend2', 'gcsicu', 'ct_sum']

	classes = ['fssmental', 'fsssensory', 'fsscommun', 'fssmotor', 'fssfeeding', 'fssresp']
	for i in range(len(classes)):
		classes[i] = labels.index(classes[i])

	fss_datasets = [fss_mental_feats, fss_sensory_feats, fss_commun_feats, fss_motor_feats, fss_feeding_feats, fss_resp_feats]
	for i in range(len(fss_datasets)):
		for j in range(len(fss_datasets[i])):
			fss_datasets[i][j] = labels.index(fss_datasets[i][j])
		fss_datasets[i] = hackathon_fss_data[:, fss_datasets[i]]

	num_ppl = len(fss_datasets[0])

	new_arr = np.concatenate(fss_datasets, axis = 1)
	final_arr = np.concatenate([new_arr, hackathon_fss_data[:, classes]], axis = 1)

	str_cols = set() # columns that need to be changed from str to int
	for i in range(num_ppl):
		for j in range(len(final_arr[i])):
			if type(final_arr[i][j]) == str:
				str_cols.add(j)

	for col in str_cols:
		str_list = [] # contains the strings for each column
		for i in range(num_ppl):
			str_list.append(final_arr[i][col])

		label_encoder = LabelEncoder()
		new_vals = label_encoder.fit_transform(str_list) # converts to ints
		for i in range(num_ppl):
			final_arr[i][col] = int(new_vals[i]) # replaces the strs with ints

	for i in range(num_ppl):
		for j in range(len(final_arr[i])):
			if np.isnan(final_arr[i][j]):
				final_arr[i][j] = -1
			else:
				final_arr[i][j] = int(final_arr[i][j])

	final_features = ['age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittocathend1', 'admittoext', 'age', 'hosplos', 'admittoicudc1', 'admittocathend3', 'admittocathend2', 'admittoicpend1', 'admittoext', 'age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittoext', 'admittocathend1', 'admittocathend3', 'admittocathstart2', 'admittoint', 'admittogast', 'admittoicpstart1', 'icptype1', 'gcsed', 'admittocathstart3', 'age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos', 'gcsett_sum', 'gcsed_sum', 'gcspar_sum', 'gcseyeob_sum', 'ct_sum', 'cardiacarrest_sum', 'age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos', 'gcsett_sum', 'gcsed_sum', 'gcspar_sum', 'gcseyeob_sum', 'ct_sum', 'cardiacarrest_sum', 'age', 'hosplos', 'admittoicudc1', 'admittoext', 'admittocathend2', 'admittogast', 'admittoicpend1', 'admittocathend1', 'admittoint', 'admittotrach', 'admittocathend3', 'newtrachyn', 'admittocathstart3', 'gcsed', 'admittoicpend2', 'gcsicu', 'ct_sum', 'fssmental', 'fsssensory', 'fsscommun', 'fssmotor', 'fssfeeding', 'fssresp']

	hackathon_fss_data = pd.DataFrame(final_arr, columns = final_features)



    # User Defined Code ends here
    ##############################################################################

    return hackathon_fss_data


################################################################################
#                                 End of File
###############################################################################.
