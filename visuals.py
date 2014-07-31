import data, ad_hoc, load
import random, string
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def changes_hist(df, title, optional='', by_degree=False):
	plt.figure()
	uniques = len(df.value_counts())
	if uniques > 100:
		uniques = 100
	df.hist(by=None, ax=None, grid=True, figsize=None, normed=True,
		color='#4682B4', bins=uniques, histtype='stepfilled')
	if by_degree:
		plt.xlabel('Degree of Change')
		optional += ' by Degree'
	else:
		plt.xlabel('Number of Changes')
	plt.ylabel('Relative Frequency')
	plt.title('Histogram of ' + title + 'Changes' + optional)
	plt.tight_layout()
	save1 = string.replace(title,' ','_')
	save2 = string.replace(optional,' ','_')
	plt.savefig(save1 + 'Changes' + save2 + '.png')
	plt.close()

# data sets to look at

all = load.train_data.ALL
drop_labels = [('X','record_type'),('X','shopping_pt')]
all = all.drop(drop_labels, axis=1)

# first shopping point
first_row = all.groupby(all.index).aggregate(lambda x: x.iget(0))
# last shopping point
last_row = all.groupby(all.index).aggregate(lambda x: x.iget(-1))
# second to last shopping point
second_last = all.groupby(all.index).aggregate(lambda x: x.iget(-2))

if __name__ == '__main__':

	# PLOTS, TABLES AND EXPLORATORY ANALYSIS

	# Parson correlation tables
	xy_corr = load.minimal.corr().Y.loc['X']
	corr_html = open('correlation_table_0.html', 'w')
	corr_html.write(xy_corr.to_html())
	corr_html.close()
	
	xy_corr = xy_corr.abs().rank(ascending=False)
	xy_corr['overall'] = xy_corr.mean(axis=1).rank()
	corr_html = open('correlation_table_1.html', 'w')
	corr_html.write(xy_corr.to_html(col_space=5))
	corr_html.close()

	# Spearman correlation tables
	xy_corr = load.minimal.corr(method='spearman').Y.loc['X']
	corr_html = open('correlation_table_2.html', 'w')
	corr_html.write(xy_corr.to_html())
	corr_html.close()

	xy_corr = xy_corr.abs().rank(ascending=False)
	xy_corr['overall'] = xy_corr.mean(axis=1).rank()
	corr_html = open('correlation_table_3.html', 'w')
	corr_html.write(xy_corr.to_html(col_space=5))
	corr_html.close()

	stat = []
	value = []

	temp = first_row.drop([('X','state')], axis=1)
	for col in temp.X.columns:
		stat.append(col + ' mean')
		value.append(temp.X[col].mean())
		stat.append(col + ' STD')
		value.append(temp.X[col].std())

	summ = pd.DataFrame({'First Shopping Point Statistic': stat, 'Value': value})
	summary_html = open('summary_table_0.html', 'w')
	summary_html.write(summ.to_html())
	summary_html.close()

	# initial = data.Stats(load.train_data, name='initial')
	# temp = load.purchased.drop([('X','state'),('X','record_type')], axis=1)
	# for y in initial.classes:
	# 	initial.Y_hist(df=temp, class_name=y)
	# initial.count_missing()

	# sample = load.purchased.drop(load.drop_labels, axis=1)
	# sample = sample.ix[random.sample(sample.index, 300)]
	# for y in load.policy_classes:
	# 	ad_hoc.scatter_matrix(sample, y, y)
	
	# Examining feature changes for customers, e.g. location, group_size

	state_diff = (first_row.X.state != last_row.X.state).astype(int, copy=False)
	diff = first_row.drop([('X','state')], axis=1).\
		sub(last_row, axis='index', level=None, fill_value=None)
	diff[('X','state')] = state_diff

	diff_count = diff[diff != 0].count(axis=0)
	plt.figure()
	diff_count.div(last_row.count(), axis=0).X.\
		plot(kind='barh', linewidth=0, legend=False, color='#4682B4')
	plt.ylabel('Column')
	plt.xlabel('Relative Frequency')
	plt.title('Number of Changes in Features Between Shopping Points')
	plt.tight_layout()
	plt.savefig('feature_changes.png')

	stat = []
	value = []

	for col in diff.X.columns:
		stat.append(col + ' mean')
		value.append(diff.X[col].mean())
		stat.append(col + ' STD')
		value.append(diff.X[col].std())

	diff_summ = pd.DataFrame({'Feature Difference Statistic': stat, 'Value': value})
	summary_html = open('summary_table_1.html', 'w')
	summary_html.write(diff_summ.to_html())
	summary_html.close()

	diff_degree = diff.X - diff.X.mean()
	diff_degree = diff_degree/diff.X.std()
	changes_hist(diff_degree['cost'], 'Cost ', by_degree=True) 

	# differences between the second to last policies and the last policy

	diff_count_col = diff.Y[diff.Y != 0].count(axis=1)
	
	changes_hist(diff_count_col, 'Policy ', ' from First to Last')
	changes_hist(diff.Y.sum(axis=1), 'Policy ', ' from First to Last',
		by_degree=True)

	second_diff = second_last.drop([('X','state')], axis=1).\
		sub(last_row, axis='index')
	
	Y_changes_sum = second_diff.Y[second_diff.Y != 0].count(axis=1)
	changes_hist(Y_changes_sum, 'Second to Last Policy ')

	Y_diff_sum = second_diff.Y.sum(axis=1)
	changes_hist(Y_diff_sum, 'Second to Last Policy ', by_degree=True)
	
	def hist_values(df):
		uniques = len(df.value_counts())
		values = np.histogram(df, bins=uniques, range=None, normed=True)
		return values

	temp = hist_values(Y_changes_sum)
	right_tail = np.sum(temp[0][(temp[1] > 0)[:-1]])
	print 'Trobability of any customer purchasing the last viewed policy is ' + \
		str((1-right_tail)*100) + '%'

	# temp = hist_values(Y_diff_sum)
	# right_tail = np.sum(temp[0][(temp[1] > 0)[:-1]])
	# left_tail = np.sum(temp[0][(temp[1] < 0)[1:]])
	# print right_tail, left_tail