#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np
from ecdf import ecdf
from bootstrap import draw_bs_reps


def main():
	''' main function'''

	file_name = 'data.csv'
	df = pd.read_csv(file_name)
	overlap = df.loc[:,'Partial Overlap']
	overlap_avg = overlap.mean()
	overlap_sum = overlap.sum()

	prior = df.loc[:,'Prior']
	prior_avg = prior.mean()
	prior_sum = prior.sum()

	# Take 10,000 samples out of the binomial distribution: n_defaults
	overlapping = np.random.binomial(32, overlap_avg, size=10000)
	prior = np.random.binomial(32, prior_avg, size=10000)	
	guessing = np.random.binomial(32, 0.5, size=10000)
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


def hypothesis_test(predict, guess):
	''' Do a hypothesis test'''

	# Compute the difference in mean sperm count: diff_means
	diff_means = np.mean(predict) - np.mean(guess)

	# Compute mean of pooled data: mean_count
	mean_count = np.mean(np.concatenate((predict, guess)))

	# Generate shifted data sets
	predict_shifted = predict - np.mean(predict) + mean_count
	guess_shifted = guess - np.mean(guess) + mean_count

	# Generate bootstrap replicates
	bs_reps_predict = draw_bs_reps(predict_shifted, np.mean, size=10000)
	bs_reps_guess = draw_bs_reps(guess_shifted, np.mean, size=10000)

	# Get replicates of difference of means: bs_replicates
	bs_replicates = bs_reps_predict - bs_reps_guess

	# Compute and print p-value: p
	p = np.sum(bs_replicates >= np.mean(predict) - np.mean(guess)) / len(bs_replicates)
	print('p-value =', p)


'''def test(predict, guess)
	 another way

	perm_replicates = draw_perm_reps('''

if __name__ == '__main__':
	main()
