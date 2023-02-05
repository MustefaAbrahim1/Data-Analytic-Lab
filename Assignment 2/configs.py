import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import  BaseEstimator, TransformerMixin

from sklearn import preprocessing





def split_data(features_based, data):
	
	split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
	for train_indeces, test_indeces in split.split(data, data[features_based]):
		train_data = data.iloc[train_indeces]
		test_data = data.iloc[test_indeces]

	return train_data, test_data

def check_split_error(data, train_data, test_data, features_based):
	overall     = np.array(data[features_based].value_counts()        / len(data))
	train = np.array(train_data[features_based].value_counts() / len(train_data))
	test  = np.array(test_data[features_based].value_counts() / len(test_data))

	overall.sort()
	train.sort()
	test.sort()


	result = {'overall': overall,'train': train, 'validation': test,
	 'train_error': np.abs(overall - train),
				'test_error': np.abs(overall - test)}
	df_result = pd.DataFrame(result)
	df_result.fillna(0)
	return df_result



def Nan_handling_by_median(col):
	col = col.fillna(col.median())
	return col

def Nan_handling_by_mode(col):
	col = col.fillna(col.mode())
	return col

def counts_outliers(col):
	median = np.median(col)
	print("Column Median: ", median)

	q1 = col.quantile(0.25) 
	q3 = col.quantile(0.75) 
	print("Q1:", q1)
	print("Q3:", q3)
	print("IQR:", q3 - q1)

	outlier_lower_limit = q1 - 1.5*(q3 - q1)
	outlier_upper_limit = q3 + 1.5*(q3 - q1)

	lower_limit_outliers = col[col < outlier_lower_limit].count()
	upper_limit_outliers = col[col > outlier_upper_limit].count()
	print("lower_limit_outliers:", lower_limit_outliers)
	print("upper_limit_outliers:", upper_limit_outliers)
	print("total outliers:", upper_limit_outliers + lower_limit_outliers)
	return lower_limit_outliers, upper_limit_outliers
	
# def outlier_handling_by_median(data, col):

# 	return data

def log_scale_to_handle_outliers(data, list_of_cols, method='box-cox'):
	pt = preprocessing.PowerTransformer(method=method, standardize=False)
	log_scale = pt.fit_transform(data[list_of_cols])

	log_scale = pd.DataFrame(log_scale, columns=list_of_cols)

	
	return log_scale