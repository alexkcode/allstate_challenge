import csv, random as rand, math, numpy as np, collections 
import os.path, pdb
import scipy.spatial as spatial, scipy.stats as sp_stats
import matplotlib.pyplot as plt, matplotlib.colors as p_colors
import sklearn.preprocessing as preprocess
import sklearn.cross_validation as cv, sklearn.metrics as metrics
import pandas as pd, pandas.core.common as com
from pandas.tools.plotting import scatter_matrix
from sklearn.multiclass import OneVsRestClassifier

# -----------------------------------------------------------------------------

# STATISTICAL UTILITY

def pd_count_missing(df, df_name='', on_screen=False, write_file=True):
	"""
	Counts the number of missings values or 'NA' for each column
	then makes a txt file report and returns a pandas dataframe
	of the counts.
	
	Parameters
    ----------

	df : pandas.DataFrame
	df_name : str
	on_screen : bool
		Set this to True if you want to see the counts in the console.
	"""
	save_file = open(df_name + 'missing_count.txt', 'w')
	header = 'Column: Missing, Percentage'
	missings = []
	if on_screen: 
		print header
	elif write_file:
		save_file.write(header + '\n')
	for i in range(len(df.X.columns)):
		total = df.X.shape[0]
		count = sum(df.X.iloc[:,i].value_counts())
		missing = total - count
		percent = float(total - count) * 100/total
		column = '{0}: {1}, {2}%'.format(df.X.columns[i], missing, percent)
		missings.append([df.X.columns[i], missing, percent])
		if on_screen: 
			print column
		elif write_file:
			save_file.write(column + '\n')
	total = 'Total (non-missing): ' + str(df.X.shape[0])
	if on_screen: 
		print total
	elif write_file:
		save_file.write(total)	
	save_file.close()
	col_names = ['column','missing','percent']
	df = pd.DataFrame(missings, columns=col_names)
	return df.set_index(['column'])

# ------------------------------------------------------------------------------

# PREDICTION AND CROSS VALIDATION

def multiOut_pred(X, Y, X_pred, clf):
	# try:
	# 	clf = OneVsRestClassifier(clf)
	# except Exception, e:
	# 	raise e
	Y_pred = []
	for Y_col in Y.T:
		Y_pred.append(clf.fit(X, Y_col).predict(X_pred).astype(int))
	return np.array(Y_pred)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title('Learning Curve for' + title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Num. of Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("learning_curve.png")

# -----------------------------------------------------------------------------

# USEFUL CLASSES

class Input:
	'''
	Simply pass a CSV file to the constructor to initialize.
	Only CSV's are supported currently.
	'''

	# -----------------------------------------------------------------------------
	# PRE-PROCESSING and DATA I/O

	# PANDAS input, returns a pd.dataframe object
	def readIn_pandas(self, data_file, idx_col, read_limit, chunk_size):
		
		iter = False
		if chunk_size != None: iter = True
		
		df = pd.read_csv(data_file, sep=None, 
			header='infer', index_col=idx_col, 
			na_values=self.missing_values, na_fvalues=None, 
			true_values=None, false_values=None, delimiter=None, 
			converters=None, dtype=None, usecols=None, 
			engine='python', as_recarray=False, 
			na_filter=True, compact_ints=False, use_unsigned=False, 
			low_memory=True, buffer_lines=None, keep_default_na=True, 
			thousands=None, comment=None, decimal='.', 
			parse_dates=False, keep_date_col=False, dayfirst=False, 
			date_parser=None, nrows=read_limit, 
			iterator=iter, chunksize=chunk_size)

		# df = pd.read_csv(data_file, sep=None, \
		# 	dialect=None, compression=None, doublequote=True, \
		# 	escapechar=None, quotechar='"', quoting=0, \
		# 	skipinitialspace=False, lineterminator=None, \
		# 	header='infer', index_col=None, names=None, \
		# 	prefix=None, skiprows=None, skipfooter=None, \
		# 	skip_footer=0, na_values=missing_values, na_fvalues=None, \
		# 	true_values=None, false_values=None, delimiter=None, \
		# 	converters=None, dtype=None, usecols=None, \
		# 	engine='python', delim_whitespace=False, as_recarray=False, \
		# 	na_filter=True, compact_ints=False, use_unsigned=False, \
		# 	low_memory=True, buffer_lines=None, warn_bad_lines=True, \
		# 	error_bad_lines=True, keep_default_na=True, \
		# 	thousands=None, comment=None, decimal='.', \
		# 	parse_dates=False, keep_date_col=False, dayfirst=False, \
		# 	date_parser=None, memory_map=False, nrows=read_limit, \
		# 	iterator=iter, chunksize=read_size, verbose=False, \
		# 	encoding=None, squeeze=False, mangle_dupe_cols=True, \
		# 	tupleize_cols=False, infer_datetime_format=False)
		
		return df

	# numpy input, returns a np.ndarray
	def readIn_numpy(self, data_file, read_limit):
		if read_limit == 'None':
			f_lines = h_lines = 0
		elif read_limit < 0:
			f_lines = -read_limit
		elif read_limit >= 0:
			h_lines = read_limit

		array = np.genfromtxt(data_file, dtype=float, comments='#', \
			delimiter=',', skiprows=0, skip_header=h_lines, skip_footer=f_lines, \
			converters=None, missing='', missing_values=None, \
			filling_values=None, usecols=None, names=True, \
			excludelist=None, deletechars=None, replace_space='_', \
			autostrip=False, case_sensitive=True, defaultfmt='f%i', \
			unpack=True, usemask=False, loose=True, invalid_raise=True)

		return array

	# custom input, returns a list of lists
	# INCOMPLETE
	def readIn(self, data_file, read_limit):
		
		X_temp = []
		Y_temp = []

		data_reader = csv.reader(open(data_file))
		columns = data_reader.next()
		field_names = {s: (columns.index(s) - 1) for s in columns}
		# this sorts field_names according to value
		field_names = colllections.OrderedDict( \
			sorted(field_names.items(), key=lambda t: t[1]))

		return X_temp, Y_temp, field_names

	# add an index to allow for seperate modelling on X and Y
	def _df_XY_split(self):
		print 'Creating X, Y multi-index for columns...'
		df = self.ALL
		old_index = df.columns
		temp = []
		for i in range(len(old_index)):
			if old_index[i] not in self.classes:
				temp.append('X')
			else:
				temp.append('Y')
		self.col_multiIDX = pd.MultiIndex.from_arrays([temp, old_index])
		df.columns = self.col_multiIDX

	def _label_extract(self):
		print 'Extracting Y labels...'
		labels = []
		for y in self.classes:
			temp = self.ALL.Y.loc[:,y].values
			labels.append(np.unique(temp))
		return labels

	@classmethod
	def fromDF(self, df, Input_obj=None, classes=[], labels=[], missing_values=[]):

		"""
		Parameters
		----------
		data_file : str
			A string pointing to where you file is.
		classes : list of str
			A list of strings with the names of the classes 
			you want to predict on.
		labels : list of str
			A list of strings 
		missing_values : list
			A list of numerics or strings for PANDAS to indentify as NA.
		"""
		
		self.ALL = df
		if Input_obj == None:
			self.classes = classes
			self.labels = labels
			self.missing_values = missing_values
		else:
			self.classes = Input_obj.classes
			self.labels = Input_obj.labels
			self.missing_values = Input_obj.missing_values
		try:
			self._df_XY_split()
		except:
			print 'DataFrame is already hierarchical.'

	def __init__(self, data_file, classes=[], labels=[], 
			idx_col=0, missing_values=[], read_size=None, read_limit=None):

		"""
		Parameters
		----------
		data_file : str
			A string pointing to where you file is.
		classes : list of str
			A list of strings with the names of the classes 
			you want to predict on.
		labels : list of str
			A list of strings 
		idx_col : int
		missing_values : list
			A list of numerics or strings for PANDAS to indentify as NA.
		read_size : int
			How many lines for the reader to read in.
		read_limit : int
			How big you want chunks to be when reading in a file. This
			will make self.ALL an interable object.
		"""

		self.ALL = None
		self.col_multiIDX = []
		self.classes = classes
		self.labels = []
		self.missing_values = missing_values

		self.ALL = self.readIn_pandas(data_file, idx_col, 
			read_limit, read_size)
		self._df_XY_split()
		if not labels: 
			self.labels = self._label_extract()
		# else:
		# 	self.labels = labels

# -----------------------------------------------------------------------------
# UTILITY

	# replacing values columnwise using a dict of functions ('func_dict') 
	# where the keys are tuples such as ('X','column_name')
	# NOTE: NaN, or missing values are avoided
	# There might be a more efficient way to do this...
	def df_lambda_replace(self, func_dict, df_list=None, inplace=False):
		"""
		replacing values columnwise using a dict of functions ('func_dict') 
		where the keys are tuples such as ('X','column_name').

		Parameters
	    ----------

		func_dict : dict()
		df_list : list()
				List of dataframes. If None then it will use self.ALL.
				All dataframes in df_list must be _df_XY_split
		NOTE: NaN, or missing values will be avoided.
		"""
		df = self.ALL
		for key in func_dict:
			idx = ('X', key)
			try:
				df.loc[com.notnull(df[idx]), idx] = \
					df.loc[com.notnull(df[idx]), idx].\
					map(func_dict[key]).astype(func_dict[1])
			except:
				print 'No type specified in func_dict so float is assumed.'
				df.loc[com.notnull(df[idx]), idx] = \
					df.loc[com.notnull(df[idx]), idx].\
					map(func_dict[key]).astype(float)
		if inplace == False:
			return self

class Stats:

	def __init__(self, Input_obj, name=''):

		"""
		Parameters
		----------
		Input_obj : object of class Input
		name : str 
			Name of data for use in plotting.
		"""

		self.ALL = Input_obj.ALL
		self.classes = Input_obj.classes
		self.labels = Input_obj.labels
		self.missing_values = Input_obj.missing_values
		self.name = name

	@classmethod
	def fromDF(self, df, name='', classes=[], labels=[], missing_values=[]):

		"""
		Parameters
		----------
		data_file : str
			A string pointing to where you file is.
		classes : list of str
			A list of strings with the names of the classes 
			you want to predict on.
		labels : list of str
			A list of strings
		missing_values : list
			A list of numerics or strings for PANDAS to indentify as NA.
		name : str
			Name of data for use in plotting.
		"""

		self.ALL = df
		self.classes = classes
		self.labels = labels
		self.missing_values = missing_values
		self.name = name

	# -----------------------------------------------------------------------------
	# VISUALIZATION

	def _group_hist(self, grouped_df, ax, i):
		'''
		helper function for df_hist function
		'''
		colors = ['b','r','g','c','m','y']
		uniques = 0
		idx = 0
		for name, group in grouped_df:
			X_i = group.X.iloc[:,i].dropna()
			counts = X_i.value_counts()
			uniques = len(counts)
			# line histogram
			if uniques > 100:			
				X_freq, binEdges = np.histogram(X_i, bins=100)
			else:
				X_freq, binEdges = np.histogram(X_i, bins=uniques)
			binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
			ax.plot(binCenters, X_freq, '-', color=colors[idx], label=name)
			idx += 1
		ax.legend()

	def Y_hist(self, df=None, class_name='', save_title = ''):
		'''
		Column-by-column line histogram grouped by Y-labels.
			
		Parameters
	    ----------
	    
		df : pandas.DataFrame
			A dataframe to be used for plotting. It must be split
			into a X, Y column multi-index, s.t. df.X gives the X columns.
			If df == None then the objects 
		class_name : str
			Name of 'Y' item.
		save_title : str
			Name of the .png file plot will be saved to.
		'''
		
		num_col = df.X.shape[1]
		fig_rows = int(math.ceil(num_col/3)) + 1
		plt.figure(figsize=(fig_rows*4, 18))
		header = df.X.columns
		Y_grouped = df.groupby([('Y', class_name)])
		num_labels = len(Y_grouped.groups)
		for i in range(num_col):
			ax = plt.subplot(fig_rows, 3, i + 1)
			# for now, limited to 6 labels for 6 different colors
			self._group_hist(Y_grouped, ax, i)
			title = header[i]
			ax.set_title(title)
		plt.tight_layout()
		plt.savefig(save_title + class_name + '_label_hist.png')

	def group_hist(self, grouped, save_title = ''):
		'''
		Column-by-column line histogram grouped by Y-labels.

		Parameters
	    ----------

		grouped : pandas.DataFrame.groupby
			A dataframe to be used for plotting. It must be split
			into a X, Y column multi-index before GroupBy operation, 
			s.t. grouped.X gives the X columns. If grouped == None 
			then the objects 
		class_name : str
			Name of 'Y' item.
		save_title : str
			Name of the .png file plot will be saved to.
		NOTE: NAs are skipped in plotting process.
		'''
		if len(grouped.groups) > 6:
			error = 'Too many groups to accommodate for in current version.'
			raise ValueError(error)
		group_0 = self.ALL.X.loc[grouped.groups[0],:]
		num_col = group_0.shape[1]
		fig_rows = int(math.ceil(num_col/3)) + 1
		plt.figure(figsize=(fig_rows*4, 18))
		header = group_0.columns
		for i in range(num_col):
			ax = plt.subplot(fig_rows, 3, i + 1)
			# for now, limited to 6 labels for 6 different colors
			self._group_hist(grouped, ax, i)
			title = header[i]
			ax.set_title(title)
		plt.tight_layout()
		plt.savefig(save_title + self.name + 'group_hist.png')

	def pd_scatter_matrix(self):
		"""
		No parameters. Run on object's attributres.
		A normal scatter plot matrix using pandas.scatter_matrix. Nothing
		new here.
		"""
		class_groups = self.ALL.groupby(self.classes)
		plt.figure()
		scatter_matrix(self.ALL, alpha=0.2, figsize=(60, 60), diagonal='kde')
		plt.savefig('scatter_matrix.png')

	# using pandas dataframe histogram plotter to plot value counts by column
	def pd_col_hist(self, df, df_name='ALL_counts', bin_count=75):
		"""
		Column by column histogram.

		Parameters
	    ----------

		df : pandas.DataFrame
		df_name : str
		bin_cound : int
		"""
		plt.figure()
		#plt.subplots(nrows=10, ncols=4)
		fig_L = len(df.columns) * 2
		fig_W = len(df.columns)
		hist_plot = df.hist(grid=True, figsize=(fig_L,fig_W), normed=False, \
			color='k', alpha=0.5, bins=bin_count, histtype='stepfilled')
		#plt.autoscale(axis='x')
		plt.autoscale(axis='y', tight=True)
		plt.locator_params(tight=True, nbins=5)
		plt.ticklabel_format(style='sci', scilimits=(0,2), axis='x')
		#plt.locator_params(tight=True, nbins=7)
		plt.savefig(df_name + '_col_hist.png')

	# -----------------------------------------------------------------------------
	# STATISTICS

	def count_missing(self):
		self.missing_count = pd_count_missing(self.ALL, self.name)
		plt.figure()
		self.missing_count.drop(['percent'], axis=1).\
			plot(kind='barh', linewidth=0, legend=False, color='#AB4B52')
		plt.ylabel('Column')
		plt.xlabel('Number of NA Values')
		plt.title('Total Number of NA Values from Each Column')
		plt.tight_layout()
		plt.savefig('missings.png')

	# -----------------------------------------------------------------------------
	# UTILITY

	# replacing values columnwise using a dict of functions ('func_dict') 
	# where the keys are tuples such as ('X','column_name')
	# NOTE: NaN, or missing values are avoided
	# There might be a more efficient way to do this...
	def df_lambda_replace(self, func_dict, df_list=None):
		'''
		Replace values columnwise using a dict of functions, func_dict,
		where the keys are tuples such as like this:
		('X','column_name','type','new_column_name')
		NOTE: NaN, or missing values are avoided.
		IMPLEMENTATION NOTE: There might be a more efficient way to do this...
		'''
		if df_list == None:
			df_list = [self.ALL]
		for df in df_list:
			for key in func_dict:
				idx = ('X', key)
				try:
					df.loc[com.notnull(df[idx]), idx] = \
						df.loc[com.notnull(df[idx]), idx].\
						map(func_dict[key]).astype(func_dict[1])
				except:
					print 'No type specified in dict so float is assumed.'
					df.loc[com.notnull(df[idx]), idx] = \
						df.loc[com.notnull(df[idx]), idx].\
						map(func_dict[key]).astype(float)



class Model(Stats):

	# -----------------------------------------------------------------------------
	# MODEL EVUALATION

	# Placeholder
	def quick_cv(self, model, model_name, func = lambda x: x):
		"""
		Cross validation assessor for classification
		Regression not supported yet.
			
		Parameters
	    ----------

		model : sklearn classifier object
		model_name : str
		func : function object
			what you want to do with the data before cross validation, e.g.
			mean normalization. Keep in mind it will apply it to a dataframe
			stored in the object, so use pandas dataframe methods.
			Ex: f = lambda df: df.dropna()
		"""
		pass

	# Placeholder.
	def sklearn_cross_val(self):
		pass

	# Placeholder.
	def learning_curve(self):
		pass

	# Placeholder.
	def sklearn_grid_search(self):
		pass
