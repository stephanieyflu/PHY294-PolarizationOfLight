import numpy as np
from numpy import arange
from pandas import read_table
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
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

def findLocalMaximaMinima(arr):
 
    # Empty lists to store points of
    # local maxima and minima
    mx = []
    mn = []
    
    n = len(arr)
 
    # Checking whether the first point is
    # local maxima or minima or neither
    if(arr[0] > arr[1]):
        mx.append(0)
    elif(arr[0] < arr[1]):
        mn.append(0)
 
    # Iterating over all points to check
    # local maxima and local minima
    for i in range(1, n-1):
 
        # Condition for local minima
        if(arr[i-1] > arr[i] < arr[i + 1]):
            mn.append(i)
 
        # Condition for local maxima
        elif(arr[i-1] < arr[i] > arr[i + 1]):
            mx.append(i)
 
    # Checking whether the last point is
    # local maxima or minima or neither
    if(arr[-1] > arr[-2]):
        mx.append(n-1)
    elif(arr[-1] < arr[-2]):
        mn.append(n-1)
        
    return mx, mn

def find_max(x, y):
	'''
	Return arrays of the local maximum y-values and their corresponding x-values
	'''

	n = 5
	for i in range(n):
		max_ind, _ = find_peaks(y)
		x = x[max_ind]
		y = y[max_ind]

	return x, y

if __name__ == '__main__':
	x, y = load_data('Ex3\ex3_nosquare_1.txt')
	max_x, max_y = find_max(x, y)