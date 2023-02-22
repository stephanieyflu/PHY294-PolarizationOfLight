import numpy as np
from numpy import arange
from pandas import read_table
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme()

def load_data(path):
	'''
 	Load data in the *.txt file as x and y arrays
 	'''
	dataframe = read_table(path, header=0, index_col=False)
	data = dataframe.to_numpy()

	x, y = data[:, 0], data[:, 1]

	return x, y

def find_max(x, y):
	
	return max_x, max_y

if __name__ == '__main__':
	x, y = load_data('Ex3\ex3_nosquare_1.txt')
	plt.plot(x, y)
	plt.show()