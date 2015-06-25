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

    
    def calculate(self,recordProvider,x,y,sign):               
        #calculate mean
        
        #event input vector
        inp = np.zeros((2, 1), dtype=float)
        inp[0] = x
        inp[1] = y
        
        #vector
        d = inp - self.mean
        
        self.mean = self.mean  + d * 0.1
        
               
        
        return self
    
    def calculatePackage(self,recordProvider):
        x = self.mean[0]
        y = self.mean[1]
        
        recordProvider.drawCross( int(x),int(y))
        self.__init__()
        
        

datafile = '/home/main/edvs-dart/datfiles/DVS128-2015-06-15T16-50-30+0200-0.aedat'

calculator = Calculator()
#open file handler with data calculator and aedata file
handler = DvsDataHandler(datafile,calculator)

#set beginning event index to start with
handler.packageStart = 0 

#visualize
#click on window will show next package of events
view = DvsDataViewer(handler)



