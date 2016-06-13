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

np.set_printoptions(threshold=np.inf)

#Call the camera to take and save an image
#subprocess.call(["./../flycapture2-2.7.3.19-amd64/flycapture/bin/Triggering"])

#Converts the .pgm file to .bmp
#img = Image.open("TestImage.pgm")

#img = img.save('result.bmp')

#Converts the bitmap to an n-deminsional array.
im = Image.open("result.bmp")
#im.show()

arr = np.array(im)
smallArray = np.zeros((151,151))


#In order to find the center of the laser, we want to loop
#through the array, and set all the values that are below a certain
#threshold to zero, and then find the center of mass of the modified array

x, y  = arr.shape

#-------------------------------------------------------
#the following subtracts the background from the picture:
#for loop[s]
for i in range(0, x):
	for ii in range(0, y):
		if arr[i][ii] < 120:
			arr[i][ii] = 0

		

#-------------------------------------------------------
#The following  finds the center of mass of the array:
Y,X = ndimage.measurements.center_of_mass(arr)

#-------------------------------------------------------
#calculate the bounds for the small array

lowerX = int(X) - 75
upperX = int(X) + 75

lowerY = int(Y) - 75
upperY = int(Y) + 75

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
#show the small figure:
#fig = plt.figure(figsize=(6, 3.2))
#ax = fig.add_subplot(111)
#ax.set_title('colorMap')
#plt.imshow(smallArray)
#ax.set_aspect('equal')

#plt.colorbar(orientation='vertical')
#plt.show()
#-------------------------------------------------------

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

#mu, sigma = scipy.stats.norm.fit(xArray)

p = gaussFit(xAxis, 10000, 0, 20, 0)


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



#-------------------------------------------------------

#err = open("cameraDetect.txt", 'r')
#err = float(err.read())

print(X)
print(Y)
#print(err)

#data = open("errorfile.csv", "a")
#w = csv.writer(data)
#w.writerow( (X,Y,err) )
#data.close()

