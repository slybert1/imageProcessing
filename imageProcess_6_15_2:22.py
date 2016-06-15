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
yArray =  np.zeros(151)

for i in range(0,150):
	yArray[i] = sum(smallArray[i,:])

#function for the gaussian:

def gaussFit(arr, a, mu, sigma, b):
	return a*np.exp(-(arr-mu)**2/(2*sigma**2)) + b

xAxis = np.linspace(-75, 75, 151)

#fitting the gaussian:
popt, pcov = curve_fit(gaussFit, xAxis, xArray)

px = gaussFit(xAxis, popt[0], popt[1], popt[2], popt[3])

popty, pcovy = curve_fit(gaussFit, xAxis, yArray)

py = gaussFit(xAxis, popty[0], popty[1], popty[2], popty[3])


#calculate the center of mass of the small array:
subY, subX = ndimage.measurements.center_of_mass(smallArray)

subX = subX*um
subY = subY*um

#center of mass of the two gaussians:
gX = ndimage.measurements.center_of_mass(px)
gY = ndimage.measurements.center_of_mass(py)
gaussX = gX[0]*um
gaussY = gY[0]*um

#print the centers:

print("center of mass of subimage: ", subX, subY)
print("center of mass of gaussians: ", gaussX, gaussY)

#can we use interpolation to smooth the gaussian, finding a more accurate center?
#------------------------------------------------------
#function to interpolate the data:
f = interpolate.interp1d(xAxis, px)
fy = interpolate.interp1d(xAxis, py)

#resample with more samples:
xx = np.linspace(-75, 75, 100000)

#compute the smoothed function:
pxx = f(xx)
pyy = fy(xx)

inpX = np.argmax(pxx)
inpY = np.argmax(pyy)

inpX = inpX*subImLen/100000
inpY = inpY*subImLen/100000

print("center of mass interpolation: ", inpX, inpY)

#-------------------------------------------------------

#err = open("cameraDetect.txt", 'r')
#err = float(err.read())
#print("center:", X, Y)
#print(err)

#data = open("errorfile.csv", "a")
#w = csv.writer(data)
#w.writerow( (X,Y,err) )
#data.close()
