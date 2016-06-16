import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import optimize
from numpy import *
from pylab import *

#np.set_printoptions(threshold=np.inf)

#Call the camera to take and save an image
#subprocess.call(["./../flycapture2-2.7.3.19-amd64/flycapture/bin/Triggering"])

#Converts the .pgm file to .bmp
#img = Image.open("TestImage.pgm")

#img = img.save('result.bmp')

#Converts the bitmap to an n-deminsional array.
im = Image.open("result.bmp")

arr = np.array(im)
subArr = arr
x, y  = subArr.shape

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

lowerX = int(X) - 75
upperX = int(X) + 75
lowerY = int(Y) - 75
upperY = int(Y) + 75

smallArray = np.zeros((151,151))

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

#length of subimage in microns:
subImLen = 151*264.58

#creating the gaussian fit with the small array:

#sum of each column into rows:
xArray = np.zeros(151)

for i in range(0, 150):
	xArray[i] = sum(smallArray[:,i])

#sum of each row into columns:
yArray = np.zeros(151)

for i in range(0,150):
	yArray[i] = sum(smallArray[i,:])

#functio<F9>n for the gaussian:

guess = np.array([15000, 70, 50, 0])

def gaussFit(arr, a, mu, sigma, b):
	return a*np.exp(-(arr-mu)**2/(2*sigma**2)) + b

xAxis = np.linspace(0, 150, 151)

#fitting the gaussian:
popt, pcov = curve_fit(gaussFit, xAxis, xArray, guess)

px = gaussFit(xAxis, popt[0], popt[1], popt[2], popt[3])
print(popt)
#print(px)

popty, pcovy = curve_fit(gaussFit, xAxis, yArray, guess)

py = gaussFit(xAxis, popty[0], popty[1], popty[2], popty[3])
print(popty)
#print(py)

#calculate the center of mass of the small array:
subY, subX = ndimage.measurements.center_of_mass(smallArray)

subX = subX
subY = subY

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xAxis, px, c = 'r', label = 'Fit')
ax.scatter(xAxis, xArray)
plt.show()

#center of mass of the two gaussians:
gaussX = popt[1]
gaussY = popty[1]

#fig = plt.figure()
#plt.imshow(smallArray)
#plt.show()

#print the centers:
print("center of mass of subimage: ", subX, subY)
print("center of mass of gaussians: ", gaussX, gaussY)

#-------------------------------------------------------

#err = open("cameraDetect.txt", 'r')
#err = float(err.read())
#print("center:", X, Y)
#print(err)

#data = open("errorfile.csv", "a")
#w = csv.writer(data)
#w.writerow( (X,Y,err) )
#data.close()
