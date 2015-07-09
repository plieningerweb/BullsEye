import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from libaedata.aedata import aedata
from libaedata.aedata import aefile

class DvsDataHandler(object):
    def __init__(self, filename, calculator):
        self.calculator = calculator
        self.image = np.zeros((128, 128), dtype=float)
        self.aedata = aedata(aefile(filename, max_events=3000000))
        self.packageStart = 0

        self.packageSize = 20
        self.packageStep = 20

        self.play = True

        self.debugPackageInfo = False

    def clear_image(self):
        #multiply all pixels with 0 so they are 0
        self.image *= 0

    def doCalculation(self,x,y,sign,microtime):
        #do calcuation with new event
        if self.calculator is not None:
            self.calculator.calculate(self,x,y,sign,microtime)
    
    def drawCross(self,x,y):
        #first cross:
        width = 1
        #horizontal lilne
        self.image[x-width:x+width] -= 0.1
        
        #vertical one
        self.image[:,y-width:y+width] -= 0.1

    def drawMarker(self,x,y):
        #first cross:
        linewidth = 1
        height = 10
        width = 10

        checkMin  = 0 
        checkMax = 127

        #horizontal line
        for i in range(x-width,x+width):
            if checkMin < i < checkMax:
                self.image[i,y] -= 0.2

        for i in range(y-height,y+height):
            if checkMin < i < checkMax:
                self.image[x,i] -= 0.2

        #self.image[x-width:x+width] -= 0.1
        
        #vertical one
        #self.image[:,y-width:y+width] -= 0.1
    
    def calculatePackage(self):
        if self.calculator is not None:
            self.calculator.calculatePackage(self)

    def calculateAllPackages(self):
        c = True
        while c:
            try:
                self.next_package(self.packageSize,self.packageStep)
            except StopIteration:
                c = False

        print("all calculated")
        
    def next_package(self,step=100,frameJump=100):
        self.clear_image()
        
        data = self.aedata
        
        start = self.packageStart
        end = start+step
        if end >= len(data.x):
            end = len(data.x) - 1

        #stop if we are already at the end
        if start >= len(data.x):
            raise StopIteration

        #len(data.x)
        if(end < len(data.x)):    
            if self.debugPackageInfo:
                print("package start index",start)
                print("package timestamp start",data.ts[start])

            n = start
            c = 0
            while ((n < len(data.x)) and (c < step)):
                y = int(data.y[n])
                x = int(data.x[n])
                if(x > 127):
                    x = 127
                if(y > 127):
                    y = 127

                #only set image event to -1 if not +1
                if self.image[x,y] != +1:
                    self.image[x,y] = +1 if data.t[n] else -1
                self.doCalculation(x,y,data.t[n],data.ts[n])

                if self.debugPackageInfo:
                    print("package was ",n)
                #next package
                c += 1
                n = start + c


            start += frameJump
        
        self.packageStart = start
        self.calculatePackage()    

        return True

        
class DvsDataViewer(object):
    def __init__(self, data):
        self.data = data       
        
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        vmin, vmax = -1, 1
        self.im = plt.imshow(self.getImage(), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')

        
        #interval is time between new frames
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=150, blit=True)
        plt.show()
        

    def updatefig(self,*args):
        self.im.set_array(self.getImage())
        #important to return array here!
        #hence, the comma
        return self.im,
    
    def getImage(self):
        #only get next package if we are in play mode
        if self.data.play is True:
            #set next package in image
            self.data.next_package(self.data.packageSize, self.data.packageStep)

        #rotate image for better visualization here
        return np.rot90(self.data.image,1)
        #return self.data.image
    
    def onClick(self,event):
        self.data.play = not self.data.play
