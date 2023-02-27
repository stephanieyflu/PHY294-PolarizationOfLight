import numpy as np
from numpy import arange
import pandas as pd
from pandas import read_table
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme()

def load_data_max(path):
	'''
 	Load data in the *.txt file as x and y arrays
 	'''
	dataframe = read_table(path, index_col=False, sep=" ")
	data = dataframe.to_numpy()

	x, y = data[:, 0], data[:, 1]

	return x, y

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

	n = 3
	for i in range(n):
		max_ind, _ = find_peaks(y)
		x = x[max_ind]
		y = y[max_ind]

	return x, y

if __name__ == '__main__':
    '''
    x, y = load_data('Kath/no.txt')
    max_x, max_y = find_max(x, y)
    df = pd.DataFrame()
    df['max_x'] = max_x.tolist()
    df['max_y'] = max_y.tolist()
    np.savetxt(r'Kath/no_max.txt', df.values)
    plt.scatter(x, y)
    plt.scatter(max_x, max_y)
    plt.show()
    '''
    max_x, max_y = load_data_max('Kath/no_max.txt')
    xerr = np.ones(len(max_x)) * 0.00005
    yerr = np.ones(len(max_y)) * 0.03

    f, ax = plt.subplots()
    ax.errorbar(max_x, max_y, xerr=xerr, yerr=yerr, fmt='o', capsize=2, elinewidth=2, markersize=4)
    ax.set_title('Intensity vs. Incident Angle (No Polarizer: Unpolarized Light)')
    ax.set_xlabel('2 $\cdot$ Angle (degrees)')
    ax.set_ylabel('Intensity (volts)')
    ax.legend(["Intensity (V)"])
    f.set_size_inches(7, 5)
    plt.savefig('Figures/no.png')
    plt.show()