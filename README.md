# Fitness Tracker

## Overview
This application's purpose is to analyze data from an experiment designed to look for a relationship between elevated heart rate (HR) and [automatically reinforced problem behavior](https://www.ncbi.nlm.nih.gov/pubmed/7938787). Specifically, the experiment is looking at the overlap between elevated HR > 100 bpm and reported times of problem behavior as well as when elevated HR > 100 bpm precedes the onset of the reported times of problem behavior.

The application was tested using Python 3.6 running on Ubuntu and relies on:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)

## How it works

1. Clone the repository
2. To run from the main application from the command line, use 'chmod +x binomial_distribution.py' 
3. In the working directory, there is a file name 'data.csv' containing the experimental data
4. Executing the application will load this data into a pandas dataframe
5. The columns of 'Partial Overlap' and 'Precedes' will be sliced and their averages computed
6. The application will generate a probability mass function (pmf) based on 10,000 simulations of the binomial distribution for each case ('Partial Overlap', 'Precedes', and 'Chance')
7. The pmfs for 'Partial Overlap and Chance' and 'Precedes and Chance' are graphed using a matplotlib histogram
8. A confidence interval of 95% (alpha  = 0.05) was chosen to establish statistical significane 
9. A two-tailed test was used, so each tail required a p-value < 0.025  (alpha / 2 = 0.025)
10. The p-value was calculated by summing the area under the 'Chance' curve up the experimentally determined averages
11. This p-value was annotated on the plot
12. The poster folder contains the Latex file and associated files for generating the conference poster

## Future plans
* Re-doing experimental design to account for potential flaws in the original study
