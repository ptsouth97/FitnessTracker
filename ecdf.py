#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
	''' main function'''

	file_name = 'data.csv'
	df = pd.read_csv(file_name)
	print(df)
	data = df.loc[:,'Partial Overlap']
	print(data)
	ecdf(data)


def ecdf(data):
	"""Compute ECDF for a one-dimensional array of measurements."""

	# Number of data points: n
	n = len(data)

	# x-data for the ECDF: x
	x = np.sort(data)

	# y-data for the ECDF: y
	y = np.arange(1, n+1) / n

	if __name__ == '__main__':

		# Plot the CDF with axis labels
		_ = plt.plot(x, y, marker='.', linestyle='none')
		_ = plt.xlabel('Overlaps with problem behavior')
		_ = plt.ylabel('CDF')

		# Show the plot
		plt.show()

	return x, y


if __name__ == "__main__":
    main()
