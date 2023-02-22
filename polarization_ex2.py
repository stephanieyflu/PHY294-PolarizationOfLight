import numpy as np
from numpy import arange
from pandas import read_table
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme()

# https://machinelearningmastery.com/curve-fitting-with-python/

def sine2x(x, a, b, c):
	return a * np.square(x) + b * x + c

def sinesq2x(x, a, b):
	return a * x + b

def load_data(path1, path2, objective):
	'''
 	Load data in the *.txt file as x and y arrays
 	'''
	dataframe1 = read_table(path1, header=0, index_col=False)
	data1 = dataframe1.to_numpy()

	dataframe2 = read_table(path2, header=0, index_col=False)
	data2 = dataframe2.to_numpy()
	data2[:, 0] = data2[:, 0] + data1[-1, 0]

	x, y = np.concatenate((data1[:, 0], data2[:, 0])), np.concatenate((data1[:, 1], data2[:, 1]))

	if objective == sine2x:
		x = np.sin(2*x)
	
	if objective == sinesq2x:
		x = np.square(np.sin(2*x))

	return x, y

def fit(objective, path1, path2):
	'''
 	Fit the data to the objective function
  	Return the parameters of the fit
 	'''
	x, y = load_data(path1, path2, objective)
	popt, _ = curve_fit(objective, x, y)

	if objective == sine2x:
		a, b, c = popt
	
	if objective == sinesq2x:
		a, b, = popt
		c = None

	return x, y, a, b, c

def plot(x, y, a, b, c, objective):
	'''
 	Plot the data and the fit
 	'''
	red_chi_sq, residuals, uncertainty = get_residuals(x, y, a, b, c, objective)
	print('Reduced chi-squared:', red_chi_sq)

	x_line = arange(min(x), max(x), 0.01)

	if objective == sine2x:
		function = 'sine2x'
		y_line = objective(x_line, a, b, c)
	
	if objective == sinesq2x:
		function = 'sinesq2x'
		y_line = objective(x_line, a, b)

	f, (ax1, ax2) = plt.subplots(1, 2)
	f.set_size_inches(20, 8.5)

	uncertaintyx = 0.00005 * np.ones(len(x))

	n = 25
	ax1.errorbar(x[::n], y[::n], xerr=uncertaintyx[::n], yerr=uncertainty[::n], fmt='o', capsize=1, elinewidth=1, markersize=3)
	ax1.plot(x_line, y_line, '--', color='red', linewidth=1)
	ax1.set_title('Light Intensity vs. Sensor Position (3 Polarizers)')
	ax1.set_xlabel('Sensor Position (radians)')
	ax1.set_ylabel('Light Intensity (volts)')

	a = round(a, 2)
	b = round(b, 2)

	if objective == sine2x:
		c = round(c, 2)
		ax1.legend(["f(x) = "+str(a)+"x^2 + "+str(b)+"x + "+str(c), "Intensity (V)"])
	
	if objective == sinesq2x:
		ax1.legend(["f(x) = "+str(a)+"x + "+str(b), "Intensity (V)"])

	ax2.scatter(x[::n], residuals[::n], color='red', s=10)
	ax2.set_title('Residuals for Light Intensity vs. Sensor Position (3 Polarizers)')
	ax2.set_xlabel('Sensor Position (radians)')
	ax2.set_ylabel('Light Intensity (volts)')
	ax2.legend(['Reduced Chi-Squared: '+str(round(red_chi_sq, 2))])

	plt.savefig('Figures\ex2' + function + '.png')

	plt.show()

def get_parameters(a, b, c, objective):
	'''
 	Print the parameters of the fit
 	'''
	a = round(a, 2)
	b = round(b, 2)

	if objective == sine2x:
		c = round(c, 2)
		print("f(x) = "+str(a)+"x^2 + "+str(b)+"x + "+str(c))
	
	if objective == sinesq2x:
		print("f(x) = "+str(a)+"x + "+str(b))

def get_residuals(x, y, a, b, c, objective):
	'''
	Residuals = abs(actual_y - fit_y)
	'''
	if objective == sine2x:
		fit_y = objective(x, a, b, c)
	if objective == sinesq2x:
		fit_y = objective(x, a, b)
	
	residuals = abs(y-fit_y)

	uncertainty = 0.03 * np.ones(len(y)) # OR 0.03 uncertainty (further away from 1)

	red_chi_sq = chi_squared(y, fit_y, uncertainty, objective)

	return red_chi_sq, residuals, uncertainty

def chi_squared(y, fit_y, uncertainty, objective):
	'''
	Reduced chi-squared value
	'''
	if objective == sine2x:
		df = len(y) - 3
	if objective == sinesq2x:
		df = len(y) - 2

	chi_sq = sum(((y-fit_y)**2)/(uncertainty**2))

	return chi_sq / df

def fit_and_plot(objective, path1, path2):
	x, y, a, b, c = fit(objective, path1, path2)
	plot(x, y, a, b, c, objective)

	get_parameters(a, b, c, objective)

if __name__ == '__main__':
	path1 = 'Ex2\ex2trial4_1.txt'
	path2 = 'Ex2\ex2trial4_2.txt'
	fit_and_plot(sine2x, path1, path2)
	fit_and_plot(sinesq2x, path1, path2)