import numpy as np
from numpy import arange
from pandas import read_table
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme()

# https://machinelearningmastery.com/curve-fitting-with-python/

def cosine(x, a, b, c):
	return a * np.square(x) + b * x + c

def cosinesq(x, a, b):
	return a * x + b

def load_data(path, objective):
	'''
 	Load data in the *.txt file as x and y arrays
 	'''
	dataframe = read_table(path, header=0, index_col=False)
	data = dataframe.to_numpy()

	x, y = data[:, 0], data[:, 1]

	if objective == cosine:
		x = np.cos(x)
	
	if objective == cosinesq:
		x = np.square(np.cos(x))

	return x, y

def fit(objective, path):
	'''
 	Fit the data to the objective function
  	Return the parameters of the fit
 	'''
	x, y = load_data(path, objective)
	popt, _ = curve_fit(objective, x, y)

	if objective == cosine:
		a, b, c = popt
	
	if objective == cosinesq:
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

	if objective == cosine:
		function = 'cosine'
		y_line = objective(x_line, a, b, c)
	
	if objective == cosinesq:
		y_line = objective(x_line, a, b)
		function = 'cosinesq'

	f, (ax1, ax2) = plt.subplots(1, 2)
	f.set_size_inches(20, 8.5)

	uncertaintyx = 0.00005 * np.ones(len(x))

	n = 25
	ax1.errorbar(x[::n], y[::n], xerr=uncertaintyx[::n], yerr=uncertainty[::n], fmt='o', capsize=1, elinewidth=1, markersize=3)
	ax1.plot(x_line, y_line, '--', color='red', linewidth=1)
	ax1.set_title('Light Intensity vs. Sensor Position (2 Polarizers)')
	ax1.set_ylabel('Light Intensity (volts)')

	ax2.scatter(x[::n], residuals[::n], color='red', s=10)
	ax2.set_title('Residuals for Light Intensity vs. Sensor Position (2 Polarizers)')
	ax2.set_ylabel('Light Intensity (volts)')
	ax2.legend(['Reduced Chi-Squared: '+str(round(red_chi_sq, 2))])

	a = round(a, 2)
	b = round(b, 2)

	if objective == cosine:
		c = round(c, 2)
		legend = "$f(x) = $"+str(a)+"$x^2 + $"+str(b)+"$x + $"+str(c)
		ax1.legend([legend, "Intensity (V)"])
		xlabel = 'Sensor Position [$cos(x)$]'
		ax1.set_xlabel(xlabel)
		ax2.set_xlabel(xlabel)
	
	if objective == cosinesq:
		legend = "$f(x) = $"+str(a)+"$x + $"+str(b)
		ax1.legend([legend, "Intensity (V)"])
		xlabel = 'Sensor Position [$cos^2(x)$]'
		ax1.set_xlabel(xlabel)
		ax2.set_xlabel(xlabel)

	plt.savefig('Figures\ex1' + function + '.png')

	plt.show()

def get_parameters(a, b, c, objective):
	'''
 	Print the parameters of the fit
 	'''
	a = round(a, 2)
	b = round(b, 2)

	if objective == cosine:
		c = round(c, 2)
		print("f(x) = "+str(a)+"x^2 + "+str(b)+"x + "+str(c))
	
	if objective == cosinesq:
		print("f(x) = "+str(a)+"x + "+str(b))

def get_residuals(x, y, a, b, c, objective):
	'''
	Residuals = abs(actual_y - fit_y)
	'''
	if objective == cosine:
		fit_y = objective(x, a, b, c)
	if objective == cosinesq:
		fit_y = objective(x, a, b)
	
	residuals = abs(y-fit_y)

	uncertainty = 0.03 * np.ones(len(y)) # OR 0.03 uncertainty because 0.05/2 = 0.025 = 0.03

	red_chi_sq = chi_squared(y, fit_y, uncertainty, objective)

	return red_chi_sq, residuals, uncertainty

def chi_squared(y, fit_y, uncertainty, objective):
	'''
	Reduced chi-squared value
	'''
	if objective == cosine:
		df = len(y) - 3
	if objective == cosinesq:
		df = len(y) - 2
	
	chi_sq = sum((np.square(y-fit_y))/np.square(uncertainty))

	return chi_sq / df

def fit_and_plot(objective, path):
	x, y, a, b, c = fit(objective, path)
	plot(x, y, a, b, c, objective)

	get_parameters(a, b, c, objective)

if __name__ == '__main__':
	path = 'Ex1\ex1trial6.txt'
	fit_and_plot(cosine, path)
	fit_and_plot(cosinesq, path)