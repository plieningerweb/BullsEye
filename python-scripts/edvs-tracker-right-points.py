#!/usr/bin/env python
# -*- Mode: Python; tab-width: 4; coding: utf8 -*-
# coding: utf-8
#vim python identation
#http://stackoverflow.com/questions/9718405/python-indentation-borked

from dvslib import DvsDataViewer
from dvslib import DvsDataHandler

import scipy.optimize as optimize
import numpy as np

class Calculator():
    def __init__(self):
        #mean (x,y)
        self.mean = np.zeros((2, 1), dtype=float)
        self.mean[0] = 20
        self.mean[1] = 30

        #make list with 15 elements
        self.events = [0] * 30 
        self.track = False
        self.trackStop = False
        self.highEventCount = 0
        self.trackCount = 0

        self.crosses = []

        self.milliSec = 0

        self.debugEvents = False 
        self.debugEventsStartStop = False 
        self.debugEventsMean = False
        self.debugEventsTimestamp = False 


    def pauseAndConfirmOutput(self):
        a = raw_input("Continue by pressing enter")
    
    def calculate(self,recordProvider,x,y,sign,microtime):               
        #calculate mean
        

        #remove last event at end of list
        self.events.pop()
        #insert new event at top of list
        self.events.insert(0,microtime)
        if self.debugEvents:
            print("events are",self.events)

        #check now if last len(self.events) events were in last 5us
        diff = microtime - self.events[-1]
        if self.debugEventsStartStop:
            print("now - 15 event",self.events[-1],"now",microtime, "diff in us", diff)
        if diff < 5000: 
            self.highEventCount += 1
            if self.highEventCount > 30:
                self.highEventCount = 30

            if self.debugEventsStartStop:
                print("found dart start", "highEventCount is",self.highEventCount)

            if self.highEventCount >= 30 and not self.track:
                self.track = True
                if self.debugEventsStartStop:
                    print("start tracking dart")
                    self.pauseAndConfirmOutput()
        else:
            self.highEventCount -= 1
            #min is zero
            if self.highEventCount < 0:
                self.highEventCount = 0

            if self.debugEventsStartStop:
                print("found dart stop")

            if self.track is True and self.highEventCount == 0:
                self.track = False 
                if self.debugEventsStartStop:
                    print("stop tracking dart")
                    self.pauseAndConfirmOutput()

        #if self.track and not self.trackStop and sign == 1:
        if self.track and sign == 1:
            self.trackCount += 1

            #event input vector
            inp = np.zeros((2, 1), dtype=float)
            inp[0] = x
            inp[1] = y
            
            #vector
            d = inp - self.mean

            if self.debugEventsMean:
                print("mean is",self.mean,"distance is",d)
            
            #if just started tracking, converge fast
            if self.trackCount < 60:
                if self.debugEventsMean:
                    print("low trackCount, move fast")
                change = d * 0.05

            else:
                if self.debugEventsMean:
                    print("high trackCount, move slow")
                #only change slowly and ignore far away events
                ab = np.sum(np.absolute(d))
                if ab == 0:
                    ab = 1

                inv = (1 /(ab*ab))
                if self.debugEventsMean:
                    print("change abs",ab,"inverse",inv)
                #max inv can be 1, otherwise we will move to far (and miss the event)
                if inv > 1:
                    inv = 1
                change = inv * d

            if self.debugEventsMean:
                print("change mean:",change)
            self.mean = self.mean + change

            #add markers every 1ms:
            #check if same millisecond as before
            millitime = self.microToMillisec(microtime)
            if self.milliSec != millitime:
                if self.debugEventsTimestamp:
                    print("new millisecond",millitime)
                self.milliSec = millitime
                
                self.crosses.append((self.mean,microtime,millitime))
        
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
        
        
        

#more stuff to calculate result points


def getMatches(markers1,markers2):
    #both files have same timestamps (snychronzied recording)
    matches = []
    for m in markers1:
        for n in markers2:
            if m[2] == n[2]:
                matches.append([m[0],n[0],m[1],m[2]])

    return matches

def checkMatchesOkay(p):
    '''check if there is really only one match per millisecond'''
    time = []
    for i in p:
        time.append(i[3])

    lena = len(time)
    lenUnique = len(set(time))
    if lena != lenUnique:
        raise Exception("error with matches: more than on match per millisecond! not unique: {}, unique: {}".format(lena,lenUnique))



#convert to 3D point in space
def convert_px_to_space_3D(threeD_points_pixels):
    """Converts pixel values of two cameras to 3D coordinates.
Input: List of points with pixels values, each a tuple:
    point: ([R.xp,R.yp],[L.xp,L.yp],microtime,millitime)
Output: Numpy array of points with space values, each a array:
    point: (x,y,z,z_prime,microtime,millitime)"""
    #settings
    angle_per_px = 0.36 #degrees
    width_px = 128
    delta = 45 #degrees
    d = 130 #cm
    H = d/2
    
    angle_per_px *= np.pi/180.  #convert to radian
    delta *= np.pi/180.                 #convert to radian
    
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
        threeD_points_space.append([x[0],y[0],z[0],z_prime[0],point[2],point[3]])
    
    return np.array(threeD_points_space)

def estimate_hit_point(threeD_points_space):
    """Input: Numpy array of points with space values, each a array:
    point: (x,y,z,z_prime,microtime,millitime)"""
    #settings
    D = 180 #cm - distance between cameras and dart board
    
    def fit_x_over_z(z, x_a, m):
        return x_a + m*z
    def fit_y_over_z(z, y_a, b, a):
        return y_a + b*z + a*z*z
    x = threeD_points_space[:,0]
    y = threeD_points_space[:,1]
    #average z-values and transform to dart board coordinates
    z = (threeD_points_space[:,2]+threeD_points_space[:,3])/2 + D
    result_fit_x_over_z = optimize.curve_fit(fit_x_over_z, z, x)
    x_a = result_fit_x_over_z[0][0]
    result_fit_y_over_z = optimize.curve_fit(fit_y_over_z, z, y)
    y_a = result_fit_y_over_z[0][0]
    print "x_a = {}, y_a = {}".format(x_a,y_a)

    return (x_a,y_a,0)


def estimate_hit_point_via_time(threeD_points_space):
    """Input: Numpy array of points with space values, each a array:
    point: (x,y,z,z_prime,microtime,millitime)"""
    #settings
    D = 180 #cm - distance between cameras and dart board
    g = -981 #cm/s^2 - acceleration due to gravity
    g *= 0.5 #because s = 0.5*g*t*t
    
    
    x = threeD_points_space[:,0]
    y = threeD_points_space[:,1]
    #average z-values and transform to dart board coordinates
    z = (threeD_points_space[:,2]+threeD_points_space[:,3])/2 + D
    #rescale time to first value in data
    t = threeD_points_space[:,4] - threeD_points_space[0,4]
    
    def fit_x_over_t(t,x_0,v_x):
        return x_0 + v_x * t
    def fit_y_over_t(t,y_0,v_y):
        return y_0 + v_y * t + g * t * t
    def fit_z_over_t(t,z_0,v_z):
        return z_0 + v_z * t
    
    result_fit_x_over_t = optimize.curve_fit(fit_x_over_t, t, x)
    x_0 = result_fit_x_over_t[0][0]
    v_x = result_fit_x_over_t[0][1]
    
    result_fit_y_over_t = optimize.curve_fit(fit_y_over_t, t, y)
    y_0 = result_fit_y_over_t[0][0]
    v_y = result_fit_y_over_t[0][1]
    
    result_fit_z_over_t = optimize.curve_fit(fit_z_over_t, t, z)
    z_0 = result_fit_z_over_t[0][0]
    v_z = result_fit_z_over_t[0][1]
    
    t_a = - z_0 / v_z
    x_a = fit_x_over_t(t_a,x_0,v_x)
    y_a = fit_y_over_t(t_a,y_0,v_y)
    print "x_a = {}, y_a = {}".format(x_a,y_a)

    return (x_a,y_a,0)

def threeD_points_estimateplane(data):
    def fix_plane(x,a,b,c):
        return a + b*x[0] + c*x[1]

    A = np.column_stack((data[:,0], data[:,1], np.ones(data[:,0].size)))
    c, resid,rank,sigma = np.linalg.lstsq(A,data[:,2])

    print("plane is",c)

    #z = ax + by +c
    a = c[0]
    b = c[1]
    c = c[2]
    point  = np.array([0.0, 0.0, c])
    normal = np.array(np.cross([1,0,a], [0,1,b]))
    d = -point.dot(normal)

    print("point,normal,d",point,normal,d)


def threeD_points_plot3D(x,y,z,more=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import itertools
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z,x,y)

    ax.set_xlabel('z Label')
    ax.set_zlabel('y Label')
    ax.set_ylabel('x Label')

    ax.set_xlim(0,200)
    ax.set_ylim(-100,100)
    ax.set_zlim(-100,100)

    colors = ['red','green','gray']
    colors = itertools.cycle(["r", "orange", "g"])
    for i in more:
        #first one is z achis
        ax.scatter(i[2],i[0],i[1], color=next(colors) )

    plt.show()

def threeD_points_plot2D(x,y):
    plt.scatter(x, y, alpha=0.5)
    plt.show()

def transformIntoDartKOS(data):
    D = 180 #cm - distance between cameras and dart board
    z = (data[:,2]+data[:,3])/2 + D
    y = -(data[:,1])
    x = data[:,0]
    return (x,y,z)



#file: ../ITQ-2015-06-25/rechts-02.aedat
#interesting events in file are from 5200 to 7960 
#dart is coming from left and going on the upper half throuh the window
#time from 314.37 to 314.49 seconds

#datafile = '../ITQ-2015-06-25/rechts-02.aedat'
#datafile = '../testdata-2015-07-02/sync33-calibrateandhit-slave.aedat'
datafile = '../FinalSetup-2015-07-06/DVS128-2015-07-06T19-58-41+0200-0336-0.aedat'
datafile = '../FinalSetup-2015-07-06/DVS128-2015-07-06T20-12-54+0200-0336-0.aedat'

calculator = Calculator()
#open file handler with data calculator and aedata file
handler = DvsDataHandler(datafile,calculator)

#set beginning event index to start with
#handler.packageStart = 1799795 
handler.packageStart = 0

handler.packageSize = 300
handler.packageStep = 300
handler.debugPackageInfo = False

#configure what animations you want to see
#to pause animation, click on it

#show camera1 dart animation
showView1 = False 
#show camera2 dart animation
showView2 = False

#compare both results in two windows
showView3 = False


if showView1:
    view = DvsDataViewer(handler)
else:
    handler.calculateAllPackages()


#left side
#fisrt event: 1648
#last event: 3742
#datafile2 = '../ITQ-2015-06-25/links-02.aedat'
#datafile2 = '../testdata-2015-07-02/sync33-calibrateandhit-master.aedat'
datafile2 = '../FinalSetup-2015-07-06/DVS128-2015-07-06T19-58-38+0200-0123-0.aedat'
datafile2 = '../FinalSetup-2015-07-06/DVS128-2015-07-06T20-12-56+0200-0123-0.aedat'

calculator2 = Calculator()
#open file handler with data calculator and aedata file
handler2 = DvsDataHandler(datafile2,calculator2)

#set beginning event index to start with
#handler2.packageStart = 2864484 
handler2.packageStart = 0

handler2.packageSize = 300
handler2.packageStep = 300

#visualize
#click on window will show next package of events
if showView2:
    view2 = DvsDataViewer(handler2)
else:
    handler2.calculateAllPackages()

    
if showView3 is True:
    viewBoth = DvsDataViewerBoth(handler,handler2)

#get markers of both
print("markers1",calculator.crosses)
print("markers2",calculator2.crosses)

matches = getMatches(calculator.crosses,calculator2.crosses)
checkMatchesOkay(matches)
#matches contain: tuple of mean, microtime, millitime

#remove some matches from beginning and end, because there the tracking is not really exact
matches = matches[30:-20]

threeD_points_space = convert_px_to_space_3D(matches)
#print threeD_points_space

print("number of matches",len(matches))
print("number of 3d points",len(threeD_points_space[:,0]))

print "\nEstimate 1, only fit: "
estimate1 = estimate_hit_point(threeD_points_space)
print "\nEstimate 2, via time: "
estimate2 = estimate_hit_point_via_time(threeD_points_space)

#plot z point over time
#threeD_points_plot2D(threeD_points_space[:,5],threeD_points_space[:,2])

#point x point over time
#threeD_points_plot2D(threeD_points_space[:,5],threeD_points_space[:,0])

#point y point over time
#threeD_points_plot2D(threeD_points_space[:,5],-threeD_points_space[:,1])

#point 3d points in 3d diagram
#threeD_points_plot3D(threeD_points_space[:,2],threeD_points_space[:,0],threeD_points_space[:,1])

#point 3d points in 3d diagram
t = transformIntoDartKOS(threeD_points_space)
threeD_points_plot3D(t[0],t[1],t[2],[(0,-50,0),estimate1,estimate2])

#not working!
#threeD_points_estimateplane(threeD_points_space)


