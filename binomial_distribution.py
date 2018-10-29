#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np


def main():
	''' main function'''

	# Read the data file and assign to a pandas dataframe
	file_name = 'data.csv'
	df = pd.read_csv(file_name)

	# Slice the column containing 'Partial Overlap' values
	overlap = df.loc[:,'Partial Overlap']
	overlap_avg = overlap.mean()
	overlap_sum = overlap.sum()

	# Slice the column containing 'Prior' values
	prior = df.loc[:,'Prior']
	prior_avg = prior.mean()
	prior_sum = prior.sum()

	# Take 10,000 samples out of the binomial distribution for each case
	overlapping = np.random.binomial(32, overlap_avg, size=10000)
	prior = np.random.binomial(32, prior_avg, size=10000)	
	guessing = np.random.binomial(32, 0.5, size=10000)

	# Call function to plot the probability mass functions
	pmf(overlapping, overlap_sum, prior, prior_sum, guessing)


def pmf(overlap, o_sum, predict, p_sum, guess):
	'''probability mass function'''

	###################### PLOT GRAPH 1 #####################################

	# Compute bin edges: bins
	bins = np.arange(0, max(guess) + 1.5) - 0.5

	# Generate histograms
	values, bins, _ = plt.hist(guess, normed=True, bins=bins, label='chance')
	p_area = (sum(values[0:(p_sum-1)])+sum(values[(p_sum+1):(len(bins)-1)]))/sum(values)
	plt.hist(predict, normed=True, bins=bins, alpha=0.5, label='precedes')

	# Set margins
	plt.margins(0.02)

	# Label axes
	_ = plt.xlabel('Number of problem behaviors preceded by elevated HR')
	_ = plt.ylabel('probability')
	_ = plt.title('PMF for Elevated HR Precedes Problem Behavior')
	_ = plt.axvline(predict.mean(), color='k', linestyle='dashed', linewidth=1)
	_ = plt.axvline(guess.mean(), color='k', linestyle='dashed', linewidth=1)
	_ = plt.annotate('p='+str(p_area), (0,0.08))

	# Make a legend
	plt.legend(loc='upper left')
	
	# Show the plot
	plt.savefig('Predict.png')
	plt.show()
	plt.close()

	####################### PLOT GRAPH 2 #####################################

	# Compute bin edges: bins
	bins = np.arange(0, max(guess) + 1.5) - 0.5
	
	# Generate histograms
	values, bins, _ = plt.hist(guess, normed=True, bins=bins, label='chance')
	
	o_area = (sum(values[0:10])+sum(values[(o_sum-1):(len(bins)-1)]))/sum(values)

	plt.hist(overlap, normed=True, bins=bins, alpha=0.5, label='overlap')

	# Set margins
	plt.margins(0.02)

	# Label axes
	_ = plt.xlabel('Number of problem behaviors overlapping with elevated HR')
	_ = plt.ylabel('probability')
	_ = plt.title('PMF for Elevated HR Overlaps with Problem Behavior')
	_ = plt.axvline(overlap.mean(), color='k', linestyle='dashed', linewidth=1)
	_ = plt.axvline(guess.mean(), color='k', linestyle='dashed', linewidth=1)
	_ = plt.annotate('p='+str(o_area), (0,0.08))

	# Make a legend
	plt.legend(loc='upper left')

	# Show the plot
	plt.savefig('Overlap.png')
	plt.show()


if __name__ == '__main__':
	main()
