import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import optimize
from numpy import *
from pylab import *

#Converts the bitmap to an n-deminsional array.
im = Image.open("result0.bmp")
#im.show()

arr = np.array(im)

#In order to find the center of the laser, we want to loop
#through the array, and set all the values that are below a certain
#threshold to zero, and then find the center of mass of the modified array

x, y  = arr.shape
#-------------------------------------------------------
#The following  finds the center of mass of the array:
#Y,X = ndimage.measurements.center_of_mass(arr)
#-------------------------------------------------------

#creating the gaussian fit with the small array:

#sum of each column into rows:
xArray = np.zeros(1200)

for i in range(0, 1199):
	xArray[i] = sum(arr[:,i])

#sum of each row into columns:
yArray =  np.zeros(960)

for i in range(0, 959):
	yArray[i] = sum(arr[i,:])

#function for the gaussian:

def gaussFit(arr, a, mu, sigma, b):
	return a*np.exp(-(arr-mu)**2/(2*sigma**2)) + b

xAxis = np.linspace(0, 1199, 1200)

#fitting the guassian:

popt, pcov = curve_fit(gaussFit, xAxis, xArray)

print(popt)

y = gaussFit(xAxis, popt[0], popt[1], popt[2], popt[3])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(xAxis, p, c = 'b', label = 'gauss')
ax.plot(xAxis, y, c = 'r', label = 'Fit')
ax.scatter(xAxis, xArray)
plt.show()

