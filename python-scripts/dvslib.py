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

    def clear_image(self):
        #multiply all pixels with 0 so they are 0
        self.image *= 0

    def doCalculation(self,x,y,sign):
        #do calcuation with new event
        if self.calculator is not None:
            self.calculator.calculate(self,x,y,sign)
    
    def drawCross(self,x,y):
        #first cross:
        width = 1
        #horizontal lilne
        self.image[x-width:x+width] = -0.5
        
        #vertical one
        self.image[:,y-width:y+width] = -0.5
    
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
            xp = data.x[start:end]
            yp = data.y[start:end]
            tp = data.t[start:end]
            timestamps = data.ts[start:end]

            print("package start index",start)
            print("package timestamp start",timestamps[0])

            for i in range(len(xp)):
                #print("i",i,start+i,"y",yp[i],xp[i],tp[i])
                y = int(yp[i])
                x = int(xp[i])
                if(x > 127):
                    x = 127
                if(y > 127):
                    y = 127
                #x = 127 - x
                self.image[x,y] = +1 if tp[i] else -1
                #print("Y,X IS",y,x,self.image[y,x])
                #self.doCalculation(x,y,tp[i])

                #maybe consider time
                #offset = time.time() - t

                
            start += frameJump
            self.image = np.rot90(self.image,1)
            
            #iterate over points and do calculation
            #todo: not very fast
            it = np.nditer(self.image, flags=['multi_index'])
            while not it.finished:
                if(it[0] != 0):
                    self.doCalculation(it.multi_index[0],it.multi_index[1],it[0])
                    #print "%d <%s>" % (it[0], it.multi_index)
                it.iternext()
                
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
        return self.data.image
    
    def onClick(self,event):
        #on every click on the plot, continue one frame
        self.data.next_package(20,20)
