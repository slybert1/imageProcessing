import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from PIL import Image
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import optimize
from numpy import *
from pylab import *

#Call the camera to take and save an image
subprocess.call(["./../../flycapture2-2.7.3.19-amd64/flycapture/bin/Triggering"])

open("gvar.txt", 'a').close()

if os.stat("gvar.txt").st_size == 0:
	gvar = 0
else:
	var = open("gvar.txt", 'r')
	gvar = float(var.read())
	var.close


#Converts the .pgm file to .bmp
img = Image.open("TestImage.pgm")

img = img.save("result" + str(gvar) + ".bmp")

#Converts the bitmap to an n-deminsional array.
im = Image.open("result" + str(gvar) + ".bmp")

arr = np.array(im)
subArr = arr
x, y  = subArr.shape

gvar = gvar + 1
var = open("gvar.txt", 'w')
var.write('%d' % gvar)
var.close()

#-------------------------------------------------------
#the following subtracts the background from the picture:

for i in range(0, x):
	for ii in range(0, y):
		if subArr[i][ii] < 120:
			subArr[i][ii] = 0

#-------------------------------------------------------
#The following  finds the center of mass of the array:
Y,X = ndimage.measurements.center_of_mass(subArr)

#arr is the original image array, subArr the original with a subtracted background

#-------------------------------------------------------
#calculate the bounds for the small array

lowerX = int(X) - 125
upperX = int(X) + 125
lowerY = int(Y) - 125
upperY = int(Y) + 125

smallArray = np.zeros((251,251))

#now make the picture small:
m = 0
n = 0

for j in range(lowerY, upperY):
        for jj in range(lowerX, upperX):
                smallArray[m][n] = arr[j][jj]
		n = n+1
	m = m+1
	n = 0
#-------------------------------------------------------

#the number of microns per pixel:
um = 264.58

#creating the gaussian fit with the small array:

#sum of each column into rows:
xArray = np.zeros(251)

for i in range(0, 250):
	xArray[i] = sum(smallArray[:,i])

#sum of each row into columns:
yArray = np.zeros(251)

for i in range(0,250):
	yArray[i] = sum(smallArray[i,:])

#function for the gaussian:

guess = np.array([15000, 70, 50, 0])

def gaussFit(arr, a, mu, sigma, b):
	return a*np.exp(-(arr-mu)**2/(2*sigma**2)) + b

xAxis = np.linspace(0, 250, 251)

#fitting the gaussian:
popt, pcov = curve_fit(gaussFit, xAxis, xArray, guess)

px = gaussFit(xAxis, popt[0], popt[1], popt[2], popt[3])

popty, pcovy = curve_fit(gaussFit, xAxis, yArray, guess)

py = gaussFit(xAxis, popty[0], popty[1], popty[2], popty[3])

#calculate the center of mass of the small array:
subY, subX = ndimage.measurements.center_of_mass(smallArray)

smSubX = subX*um
smSubY = subY*um

#center of mass of the two gaussians:
gaussX = popt[1]*um
gaussY = popty[1]*um

#in reference to the large pictures:

fgaussX = (X+popt[1]-251/2)*um
fgaussY = (Y+popty[1]-251/2)*um

mX = X*um
mY = Y*um

#print the centers:
print("center of mass of subimage: ", subX, subY)
print("center of mass of gaussians: ", gaussX, gaussY)

i#-------------------------------------------------------

err = open("cameraDetect.txt", 'r')
err = float(err.read())
print("center:", X, Y)
print(err)

data = open("errorfile.csv", "a")
w = csv.writer(data)
w.writerow( (fgaussX, fgaussY, mX, mY, gaussX, gaussY, subX, subY, err) )
data.close()

