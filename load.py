import data
import collections, time, random as rand

import pandas as pd, numpy as np

import scipy.spatial as sp, scipy.interpolate as interp

# ------------------------------------------------------------------------------

# various files needed for training and prediction
training_file = 'train.csv'
prediction_file = 'test_v2.csv'
states_file = 'state_latlon.csv'

# the 'Y' variables to be predicted and trained on
policy_classes = ['A','B','C','D','E','F','G']

# creating feature data based on distance from socioeconomic centers
states = {}
T_dist = {}				# dist from Texas
E_dist = {}				# dist from New York
W_dist = {}				# dist from California
state_loc_data = open(states_file, 'r')
states_col_names = state_loc_data.readline()
for line in state_loc_data:
	state_data = line.split(',')
	states[state_data[0]] = [float(state_data[1]), float(state_data[2])]
for s in states:
	E_dist = {s: sp.distance.euclidean(states['NY'], states[s]) for s in states}
	W_dist = {s: sp.distance.euclidean(states['CA'], states[s]) for s in states}
	S_dist = {s: sp.distance.euclidean(states['TX'], states[s]) for s in states}
state_loc_data.close()

# dictionary to convert car_val to numeric values
car_val = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, \
	'i': 8, 'j': 9, 'k': 10, 'l': 11}

# ------------------------------------------------------------------------------

# LOADING DATA

train_data = data.Input(training_file, classes=policy_classes,
	idx_col=0, missing_values=['','NA','NaN'], read_limit=None)

test_data = data.Input(prediction_file, classes=policy_classes,
	idx_col=0, missing_values=['','NA','NaN'], read_limit=None)

# PREPROCESSING

"""
time -> hours and minutes gets converted to a float numerical of hours
car_value -> straight numeric conversion
location -> the numbers were offset by about 10,000 so I removed that offset
"""
func_dict = {'time' : lambda x: float(x[1]) + float(x[0])/60,
			'car_value' : lambda x: car_val[x],
			'location' : lambda x: x - 10000}

train_data = train_data.df_lambda_replace(func_dict)

# Segregating and Grouping

drop_labels = ['record_type',
			'shopping_pt',
			'state',]
drop_labels = [('X', a) for a in drop_labels]

purchased = train_data.ALL.loc[train_data.ALL[('X','record_type')]==1]
minimal = train_data.ALL.drop(drop_labels)
by_C_previous = train_data.ALL.groupby([('X','C_previous')])
by_home = train_data.ALL.groupby([('X','homeowner')])
by_state = train_data.ALL.groupby([('X','state')])

print 'There are {0} customer_IDs.'.format(purchased.shape[0])