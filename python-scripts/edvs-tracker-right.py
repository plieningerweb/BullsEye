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

        if self.track:
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

            #add markers every n-th time
            if (self.trackCount % 500) == 0:
                print("add marker cross")
                self.crosses.append(self.mean)
                print("markers are",self.crosses)
        
        return self
    
    def calculatePackage(self,recordProvider):
        x = self.mean[0]
        y = self.mean[1]

        #draw cross for every item
        for c in self.crosses:
            recordProvider.drawCross( int(c[0]),int(c[1]))
        
        recordProvider.drawCross( int(x),int(y))
#        self.__init__()

        
        
        
#file: ../ITQ-2015-06-25/rechts-02.aedat
#interesting events in file are from 5200 to 7960 
#dart is coming from left and going on the upper half throuh the window
#time from 314.37 to 314.49 seconds

datafile = '../ITQ-2015-06-25/rechts-02.aedat'

calculator = Calculator()
#open file handler with data calculator and aedata file
handler = DvsDataHandler(datafile,calculator)

#set beginning event index to start with
handler.packageStart = 5250

handler.packageSize = 200
handler.packageStep = 200

#visualize
#click on window will show next package of events
view = DvsDataViewer(handler)



#idea one:
#find active area with >=15 events in 10ms
#set mode to track dart
#now track the dart with a mean, where more distant events are weighted less strongo
#calculate a "moving average of the centroid"
