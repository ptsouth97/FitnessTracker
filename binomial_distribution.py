#!/usr/bin/python3

import matplotlib.pyplot as plt	
import numpy as np
from ecdf import ecdf
from bootstrap import draw_bs_reps


def main():
	''' main function'''

	# Take 10,000 samples out of the binomial distribution: n_defaults
	predicting = np.random.binomial(32, 0.65625, size=10000)	
	print(predicting)
	guessing = np.random.binomial(32, 0.5, size=10000)
	print(guessing)
	pmf(predicting, guessing)
	hypothesis_test(predicting, guessing)


def binom():
	''' binomial distribution'''


	# Compute CDF: x, y
	x, y = ecdf(n_defaults)

	# Plot the CDF with axis labels
	_ = plt.plot(x, y, marker='.', linestyle='none')
	_ = plt.xlabel('Overlaps with problem behavior')
	_ = plt.ylabel('CDF')

	# Show the plot
	plt.show()


def pmf(predict, guess):
	'''probability mass function'''

	# Compute bin edges: bins
	bins = np.arange(0, max(guess) + 1.5) - 0.5

	# Generate histogram
	plt.hist(guess, normed=True, bins=bins, label='guessing')
	plt.hist(predict, normed=True, bins=bins, alpha=0.5, label='predicting')

	# Set margins
	plt.margins(0.02)

	# Label axes
	_ = plt.xlabel('number of problem behaviors correctly predicted by HR')
	_ = plt.ylabel('probability')

	# Make a legend
	plt.legend(loc='upper left')
	
	# Show the plot
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
