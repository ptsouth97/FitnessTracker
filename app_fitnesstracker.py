#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np


def main():
	''' main function for loading data, calculating metrics, and calling pmf function'''

	# Read the data file and assign to a pandas dataframe
	file_name = 'data.csv'
	df = pd.read_csv(file_name)

	# Slice the column containing 'Partial Overlap' values
	overlap = df.loc[:,'Partial Overlap']

	# Compute metrics of interest
	overlap_len = len(overlap)
	overlap_avg = overlap.mean()
	overlap_sum = overlap.sum()

	# Slice the column containing 'Precedes' values
	precedes = df.loc[:,'Precedes']

	# Compute metrics of interest
	precedes_len = len(precedes)
	precedes_avg = precedes.mean()
	precedes_sum = precedes.sum()

	# Calculate chance sum (how many occurences expected by chance)
	chance_prob = 0.05272392315
	chance_sum = len(precedes) * chance_prob

	# Take 10,000 samples out of the binomial distribution for each case
	overlapping = np.random.binomial(overlap_len, overlap_avg, size=10000)
	preceding = np.random.binomial(precedes_len, precedes_avg, size=10000)	
	chance = np.random.binomial(precedes_len, chance_prob, size=10000)

	# Call function to plot the probability mass function for OVERLAP
	pmf(overlapping, overlap_sum, chance, chance_sum, 'Overlaps' )

	# Call function to plot the probability mass function for PRECEDES
	pmf(preceding, precedes_sum, chance, chance_sum, 'Precedes')


def pmf(condition_df, condition_sum, chance_df, chance_sum, condition):
	''' Plots probability mass function along with p-value'''

	# Compute bin edges: bins
	bins = np.arange(0, max(condition_df)+1.5) - 0.5

	# Generate chance histogram
	values, bins, _ = plt.hist(chance_df, normed=True, bins=bins, label='Chance')

	# Generate condition histogram
	_ = plt.hist(condition_df, normed=True, bins=bins, alpha=0.5, label=condition)

	calc_bins = np.arange(min(chance_df), max(condition_df))

	# Calculate p-value/2 (two-tail test) for when the value is greater or less than the avg
	if condition_sum > chance_sum:
		p = sum(values[(condition_sum):(max(calc_bins))])/sum(values)

	else:
		p = sum(values[0:(condition_sum+1)])/sum(values)

	# Set margins
	_ = plt.margins(0.02)

	# Label axes
	_ = plt.xlabel('Number of successes')
	_ = plt.ylabel('probability')
	_ = plt.suptitle('PMF for Elevated HR ' +condition+ ' Problem Behavior')
	_ = plt.title('p/2='+str(p))
	_ = plt.axvline(condition_df.mean(), color='k', linestyle='dashed', linewidth=1)
	_ = plt.axvline(chance_df.mean(), color='k', linestyle='dashed', linewidth=1)
	#_ = plt.annotate('p/2='+str(p), (0,0.08))

	# Make a legend
	plt.legend(loc='upper left')
	
	# Show the plot
	name = condition+'.png'
	plt.savefig(name)
	plt.show()
	plt.close()


if __name__ == '__main__':
	main()
