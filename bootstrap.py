#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns

def main():
	df = pd.read_csv('Numerical Responses.csv', header=1)
	respondent1 = df.iloc[0][14:57]
	
	reps = draw_bs_reps(respondent1, np.mean, 10000)
	# _ = plt.hist(bs_replicates, normed=True, bins=3)
	# plt.show()

	conf_int = np.percentile(reps, [2.5, 97.5])
	print('95% confidence interval =', conf_int)


def gen_bs_sample(data):
	bs_sample = np.random.choice(data, size=len(data))

	sns.set()
	x, y = ecdf.ecdf(bs_sample)
	_ = plt.plot(x, y, marker='.', linestyle='none', color='blue', alpha=0.3, label='bs_sample')

	x_orig, y_orig = ecdf.ecdf(data)
	_ = plt.plot(x_orig, y_orig, marker='.', color='red', linestyle='none', label='original data')

	plt.margins(0.02)
	_ = plt.title('Respondent 1 - All Responses')
	_ = plt.xlabel('Survey response')
	_ = plt.ylabel('ECDF')
	plt.legend(loc='lower right')

	plt.show()

	#print(bs_sample)
	#print(data)


def bootstrap_replicate_1d(data, func):
	'''Generate bootstrap replicate of 1D data.'''
	bs_sample = np.random.choice(data, len(data))
	return func(bs_sample)


def draw_bs_reps(data, func, size=1):
	'''Draw bootstrap replicates'''

	# Initialize array of replicates
	bs_replicates = np.empty(size)	

	# Generate replicates
	for i in range(size):
		bs_replicates[i] = bootstrap_replicate_1d(data, func)
	
	return bs_replicates	


def draw_bs_pairs_linreg(x, y, size=1):
	'''Perform pairs bootstrap for linear regression'''

	# Set up array of indices to sample from
	inds = np.arange(len(x))

	# Initialize replicates
	bs_slope_reps = np.empty(size)
	bs_intercept_reps = np.empty(size)

	# Generate replicates
	for i in range(size):
		bs_inds = np.random.choice(inds, size=len(inds))
		bs_x, bs_y = x[bs_inds], y[bs_inds]
		bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

	return bs_slope_reps, bs_intercept_reps


if __name__ == '__main__':
	main()
