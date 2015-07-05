# coding: utf-8
#vim python identation
#http://stackoverflow.com/questions/9718405/python-indentation-borked

from dvslib import DvsDataViewer
from dvslib import DvsDataHandler


import numpy as np

class Calculator():
    def __init__(self):
        #mean (x,y)
        self.mean = np.zeros((2, 1), dtype=float)
        self.mean[0] = 20
        self.mean[1] = 30

        #make list with 15 elements
        self.events = [0] * 15 
        self.track = False
        self.trackCount = 0

        self.crosses = []

        self.milliSec = 0

    
    def calculate(self,recordProvider,x,y,sign,microtime):               
        #calculate mean
        

        #remove last event at end of list
        self.events.pop()
        #insert new event at top of list
        self.events.insert(0,microtime)
        print("events are",self.events)

        #check now if last 15 events were in last 5ms
        diff = microtime - self.events[-1]
        print("now - 15 event",self.events[-1],"now",microtime, "diff in us", diff)
        if diff < 5000: 
            self.track = True
            print("found dart")

        if self.track and sign == 1:
            self.trackCount += 1

            #event input vector
            inp = np.zeros((2, 1), dtype=float)
            inp[0] = x
            inp[1] = y
            
            #vector
            d = inp - self.mean

            print("mean is",self.mean,"distance is",d)
            
            #if just started tracking, converge fast
            if self.trackCount < 60:
                print("low trackCount, move fast")
                change = d * 0.05

            else:
                print("high trackCount, move slow")
                #only change slowly and ignore far away events
                ab = np.sum(np.absolute(d))
                inv = (1 /(ab*ab))
                print("change abs",ab,"inverse",inv)
                #max inv can be 1, otherwise we will move to far (and miss the event)
                if inv > 1:
                    inv = 1
                change = inv * d

            print("change mean:",change)
            self.mean = self.mean + change

            #add markers every 1ms:
            #check if same millisecond as before
            millitime = self.microToMillisec(microtime)
            if self.milliSec != millitime:
                print("new millisecond",millitime)
                self.milliSec = millitime
                
                print("add marker cross")
                self.crosses.append((self.mean,microtime))
                print("markers are",self.crosses)
        
        return self

    def microToMillisec(self,microtime):
        return int(microtime/1000)
    
    def calculatePackage(self,recordProvider):
        x = self.mean[0]
        y = self.mean[1]

        #draw cross for every item
        for c in self.crosses:
#            recordProvider.drawCross( int(c[0]),int(c[1]))
            point = c[0]
            recordProvider.drawMarker( int(point[0]),int(point[1]))
        
        recordProvider.drawCross( int(x),int(y))
#        self.__init__()

        
        
        
#file: ../ITQ-2015-06-25/rechts-02.aedat
#interesting events in file are from 5200 to 7960 
#dart is coming from left and going on the upper half throuh the window
#time from 314.37 to 314.49 seconds

datafile = '../ITQ-2015-06-25/rechts-02.aedat'
datafile = '../ITQ-2015-06-25/rechts-02.aedat'

calculator = Calculator()
#open file handler with data calculator and aedata file
handler = DvsDataHandler(datafile,calculator)

#set beginning event index to start with
handler.packageStart = 5250

handler.packageSize = 300
handler.packageStep = 300

#visualize
#click on window will show next package of events
view = DvsDataViewer(handler)



#idea one:
#find active area with >=15 events in 10ms
#set mode to track dart
#now track the dart with a mean, where more distant events are weighted less strongo
#calculate a "moving average of the centroid"

#idea to match the two cameras:
#calculate the points for each camera
#compare the points including timestamp and match the points (same time, where is my marker?)
#--> calculate 3d point
#--> curve fit parabel in 3dimensions


#left side
#fisrt event: 1648
#last event: 3742
datafile2 = '../ITQ-2015-06-25/links-02.aedat'

calculator2 = Calculator()
#open file handler with data calculator and aedata file
handler2 = DvsDataHandler(datafile2,calculator2)

#set beginning event index to start with
handler2.packageStart = 1648 

handler2.packageSize = 300
handler2.packageStep = 300

#visualize
#click on window will show next package of events
view2 = DvsDataViewer(handler2)

#view both stuff
import matplotlib.pyplot as plt
class DvsDataViewerBoth(object):
    def __init__(self, data1, data2):
        
        fig1 = plt.figure()
        vmin, vmax = -1, 1
        im1 = plt.imshow(self.getImage(data1), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')

        fig2 = plt.figure()
        im2 = plt.imshow(self.getImage(data2), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')

        plt.show()
        
    def getImage(self,data):
        #rotate image for better visualization here
        return np.rot90(data.image,1)
        #return self.data.image
    

viewBoth = DvsDataViewerBoth(view.data,view2.data)

#dart found in view2 at package 1691
#time of package: 300221859.0, 
dart2RecognizedTime = 300221859.0


#dart found in view1 at package 5305
#time of package: 314293710.0,
dart1RecognizedTime = 314293710.0

dartTimeDiff = dart2RecognizedTime - dart1RecognizedTime

#get markers of both
print("markers1",calculator.crosses)
print("markers2",calculator2.crosses)

threeD_points_pixels = []

for m in calculator.crosses:
    #print("m of markers1 is",m)
    #print("find marker with time X in markers2",m[1],m[1]+dartTimeDiff)

    for n in calculator2.crosses:
        #print("n in markers2 is",n)
        diff = abs(n[1]-(m[1]+dartTimeDiff))
        #print("diff is",diff)
        if diff < 1000:
            print("match < 1ms")
            threeD_points_pixels.append((m[0],n[0]))

#convert to 3D point in space
def convert_px_to_space_3D(threeD_points_pixels):
	"""Converts pixel values of two cameras to 3D coordinates.
Input: List of points with pixels values, each a tuple:
	point: ([R.xp,R.yp],[L.xp,L.yp])
Output: List of points with space values, each a tuple:
	point: (x,y,z,z_prime)"""
	#settings
	angle_per_px = 0.36 #degrees
	width_px = 128
	delta = 45 #degrees
	d = 120 #cm
	H = d/2
	
	angle_per_px *= np.pi/180.	#convert to radian
	delta *= np.pi/180.			#convert to radian
	
	threeD_points_space = []
	center_offset_px = (width_px / 2.) -0.5
	for point in threeD_points_pixels:
		#point: ([R.xp,R.yp],[L.xp,L.yp])
		#right camera
		gamma_r = angle_per_px * (point[0][0] - center_offset_px)
		bita_r = angle_per_px * (point[0][1] - center_offset_px)
		#left camera
		gamma_l = angle_per_px * (point[1][0] - center_offset_px)
		alpha_l = angle_per_px * (point[1][1] - center_offset_px)
		#convert to space
		d_l = d / ( 1. + (np.tan(delta-alpha_l)/np.tan(delta-bita_r)) )
		x = d_l - d
		y = np.tan(delta - alpha_l) * d_l - H
		z = np.tan(gamma_l) * np.cos(alpha_l) * d_l / np.cos(delta - alpha_l)
		z_prime = - np.tan(gamma_r) * np.cos(bita_r) * (d-d_l) / np.cos(delta - bita_r)
		#save
		threeD_points_space.append([x,y,z,z_prime])
	
	return threeD_points_space

print threeD_points_pixels

threeD_points_space = convert_px_to_space_3D(threeD_points_pixels)

print threeD_points_space

#scipy optimize curve fit:
#http://python4mpia.github.io/fitting_data/least-squares-fitting.html
