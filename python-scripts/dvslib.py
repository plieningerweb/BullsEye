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
        self.aedata = aedata(aefile(filename, max_events=100000))
        self.packageStart = 0

        self.packageSize = 20
        self.packageStep = 20

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
    
    def calculatePackage(self):
        if self.calculator is not None:
            self.calculator.calculatePackage(self)
        
    def next_package(self,step=1000,frameJump=30):
        self.clear_image()
        
        data = self.aedata
        
        start = self.packageStart
        end = start+step
        #len(data.x)
        if(end < len(data.x)):    
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

                self.image[x,y] = +1 if data.t[n] else -1
                self.doCalculation(x,y,data.t[n],data.ts[n])

                print("package was ",n)
                #next package
                c += 1
                n = start + c


            start += frameJump
        else:
            raise Exception("end of data")
        
        self.packageStart = start
        self.calculatePackage()    

        
class DvsDataViewer(object):
    def __init__(self, data):
        self.data = data       
        
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        vmin, vmax = -1, 1
        self.im = plt.imshow(self.getImage(), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')

        
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=20, blit=True)
        plt.show()
        

    def updatefig(self,*args):
        self.im.set_array(self.getImage())
        #important to return array here!
        #hence, the comma
        return self.im,
    
    def getImage(self):
        #rotate image for better visualization here
        return np.rot90(self.data.image,1)
        #return self.data.image
    
    def onClick(self,event):
        #on every click on the plot, continue one frame
        self.data.next_package(self.data.packageSize, self.data.packageStep)
