
# coding: utf-8

# In[4]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import time

class RetinaRecord(object):
    def __init__(self, filename, calculator):
        self.calculator = calculator
        self.data = open(filename)
        self.image = np.zeros((128, 128), dtype=float)
        self.test="testme"
        #thread.start_new_thread(self.update_loop, ())
        self.data_iter = self.update_loop()
        
        """
        file:
        DVS128-2015-06-15T16-50-30+0200-0.aedat
        dart pfeil flieg zwischen event 2000 und 4000 auf scheibe zu
        """
        self.aedata = aedata(aefile('/home/main/edvs-dart/datfiles/DVS128-2015-06-15T16-50-30+0200-0.aedat', max_events=100000))
        #self.aedata = aedata(aefile('/home/main/edvs-dart/datfiles/DVS128-2015-06-15T16-48-27+0200-0.aedat', max_events=1000000))
        self.packageStart = 0

    def clear_image(self):
        self.image *= 0

    def update_loop(self):
        offset = None

        for line in self.data.readlines():
            x, y, sign, t = line.strip().split()
            y = int(y)
            x = int(x)
            sign = int(sign)
            t = float(t)
            if offset is None:
                offset = time.time() - t

#            while time.time() < t + offset:
#                time.sleep(0.01)
#                pass

            self.image[y, x] = +1 if sign else -1
    
            #self.drawMarker()
        
            #i think x,y is reversed already in the file
            self.doCalculation(y,x,sign)
    
            yield self.image
        
    def doCalculation(self,x,y,sign):
        #do calcuation with new event
        if self.calculator is not None:
            self.calculator.calculate(self,x,y,sign)

    def drawMarker(self):
        #http://scipy-lectures.github.io/advanced/image_processing/
        
        #line 10 to twenty, color darker
        #self.image[10:20] = -0.5
        
        #second point in each line, therefore the second column
        #self.image[:,1] = -0.5
        self.drawCross(20,20)
        
        return self
    
    def drawCross(self,x,y):
        #first cross:
        width = 1
        #horizontal lilne
        self.image[x-width:x+width] = -0.5
        
        #vertical one
        self.image[:,y-width:y+width] = -0.5
        
        
        return self
    
    def calculatePackage(self):
        if self.calculator is not None:
            self.calculator.calculatePackage(self)
        
    def next_package2(self,eventNumber):
        #first clear image
        self.clear_image()
        
        for i in range(eventNumber):
            try:
                last = self.data_iter.next()
            except StopIteration:
                pass
            
        self.image = last
        self.calculatePackage()    
    
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
        
            
            
class DataViewer(object):
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
        #self.data.next_package(128)
        return self.data.image
    
    def onClick(self,event):
        self.data.next_package(20,20)
    
    
#retina = RetinaRecord('blinking_2.data')
#view = DataViewer(retina)

"""
fig = plt.figure()

def getImage():
    global retina
    return retina.next_package(200)

vmin, vmax = -1, 1
im = plt.imshow(getImage(), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')

def updatefig(*args):
    im.set_array(getImage())
    #important to return array here!
    #hence, the comma
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)
plt.show()
"""


# In[2]:

#from...https://raw.githubusercontent.com/darioml/python-aer/master/src/__init__.py
"""
Author: Dario ML
Program: src/__init__.py
Description: main file for python-ae
"""

from PIL import Image
import math
import numpy as np
#import scipy.io
import os,time
import matplotlib.pyplot as plt
from matplotlib import cm

class aefile(object):
    def __init__(self, filename, max_events=1e6):
        self.filename = filename
        self.max_events = max_events
        self.header = []
        self.data, self.timestamp = self.read()

    # alias for read
    def load(self):
        return self.read()

    def read(self):
        with open(self.filename, 'r') as f:
            line = f.readline()
            while line[0] == '#':
                self.header.append(line)
                if line[0:9] == '#!AER-DAT':
                    aer_version = line[9];
                current = f.tell()
                line = f.readline()


            if aer_version != '2':
                raise Exception('Invalid AER version. Expected 2, got %s' % aer_version)

            f.seek(0,2)
            numEvents = math.floor( ( f.tell() - current ) / 8 )
            
            if numEvents > self.max_events:
                print 'There are %i events, but max_events is set to %i. Will only use %i events.' % (numEvents, self.max_events, self.max_events)
                numEvents = self.max_events

            f.seek(current)

            timestamps = np.zeros( numEvents )
            data       = np.zeros( numEvents )

            for i in range(int(numEvents)):
                data[i] = int(f.read(4).encode('hex'), 16)
                timestamps[i] = int(f.read(4).encode('hex'), 16)

            return data, timestamps

    def save(self, data=None, filename=None, ext='aedat'):
        if filename is None:
            filename = self.filename
        if data is None:
            data = aedata(self)
        if ext is 'aedat':
            # unpack our 'data'
            ts = data.ts
            data = data.pack()

            with open(filename, 'w') as f:
                for item in self.header:
                    f.write(item)
                print
                print
                no_items = len(data)
                for i in range(no_items):
                    f.write(hex(int(data[i]))[2:].zfill(8).decode('hex'))
                    f.write(hex(int(ts[i]))[2:].zfill(8).decode('hex'))

    def unpack(self):
        noData = len(self.data)

        x = np.zeros(noData)
        y = np.zeros(noData)
        t = np.zeros(noData)

        for i in range(noData):

            d = int(self.data[i])

            t[i] = d & 0x1
            x[i] = 128-((d >> 0x1) & 0x7F)
            y[i] = (d >> 0x8) & 0x7F
        return x,y,t


class aedata(object):
    def __init__(self, ae_file=None):
        self.dimensions = (128,128)
        if isinstance(ae_file, aefile):
            self.x, self.y, self.t = ae_file.unpack()
            self.ts = ae_file.timestamp
        elif isinstance(ae_file, aedata):
            self.x, self.y, self.t = aedata.x, aedata.y, aedata.t
            self.ts = ae_file.ts
        else:
            self.x, self.y, self.t, self.ts = np.array([]),np.array([]),np.array([]),np.array([])

    def __getitem__(self, item):
        rtn = aedata()
        rtn.x = self.x[item]
        rtn.y = self.y[item]
        rtn.t = self.t[item]
        rtn.ts= self.ts[item]
        return rtn

    def __setitem__(self, key, value):
        self.x[key] = value.x
        self.y[key] = value.y
        self.t[key] = value.t
        self.ts[key] = value.ts

    def __delitem__(self, key):
        self.x = np.delete(self.x,  key)
        self.y = np.delete(self.y,  key)
        self.t = np.delete(self.t,  key)
        self.ts = np.delete(self.ts,  key)

    def save_to_mat(self, filename):
        raise Exception('not implemented')
#        scipy.io.savemat(filename, {'X':self.x, 'Y':self.y, 't': self.t, 'ts': self.ts})

    def pack(self):
        noData = len(self.x)
        packed = np.zeros(noData)
        for i in range(noData):
            packed[i] =  (int(self.t[i]) & 0x1)
            packed[i] += (int(128-self.x[i]) & 0x7F) << 0x1
            packed[i] += (int(self.y[i]) & 0x7F) << 0x8

        return packed

    # TODO
    # performance here can be improved by allowing indexing in the AE data.
    # For now, I expect this not to be done often
    def make_sparse(self, ratio):
        indexes = np.random.randint(0,len(self.x),math.floor(len(self.x)/ratio))
        indexes.sort()

        rtn = aedata()
        rtn.x = self.x[indexes]
        rtn.y = self.y[indexes]
        rtn.t = self.t[indexes]
        rtn.ts = self.ts[indexes]

        return rtn

    def __repr__(self):
        return "%i total [x,y,t,ts]: [%s, %s, %s, %s]" % (len(self.x), self.x, self.y, self.t, self.ts)

    def __len__(self):
        return len(self.x)

    def interactive_animation(self, step=5000, limits=(0,128), pause=0):
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        plt.show()
        ax = fig.add_subplot(111)

        start = 0
        end = step-1
        while(start < len(self.x)):
            ax.clear()
            ax.scatter(self.x[start:end],self.y[start:end],s=20,c=self.t[start:end], marker = 'o', cmap = cm.jet );
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            start += step
            end += step
            plt.draw()
            time.sleep(pause)

    def downsample(self,new_dimensions=(16,16)):
        # TODO
        # Make this cleaner
        assert self.dimensions[0]%new_dimensions[0] is 0
        assert self.dimensions[1]%new_dimensions[1] is 0

        rtn = aedata()

        rtn.ts = self.ts
        rtn.t = self.t
        rtn.x = np.floor(self.x / (self.dimensions[0] / new_dimensions[0]))
        rtn.y = np.floor(self.y / (self.dimensions[1] / new_dimensions[1]))

        return rtn

    def to_matrix(self, dim=(128,128)):
        return make_matrix(self.x, self.y, self.t, dim=dim)

def make_matrix(x, y, t, dim=(128,128)):
    image = np.zeros(dim)
    events= np.zeros(dim)

    for i in range(len(x)):
        image[y[i]-1,x[i]-1] -= t[i]-0.5
        events[y[i]-1,x[i]-1] += 1

    # http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    np.seterr(divide='ignore', invalid='ignore')

    result = 0.5+(image / events)
    result[events == 0] = 0.5
    return result

def create_pngs(data,prepend,path="",step=3000,dim=(128,128)):
    if not os.path.exists(path):
        os.makedirs(path)

    idx = 0
    start = 0;
    end = step-1;
    while(start < len(data.x)):
        image = make_matrix(data.x[start:end],data.y[start:end],data.t[start:end], dim=dim)
        img_arr = (image*255).astype('uint8')
        im = Image.fromarray(img_arr)
        im.save(path+'/'+prepend+("%05d" % idx)+".png")
        idx += 1

        start += step
        end += step

def concatenate(a_tuple):
    rtn = aedata()
    rtn.x = np.concatenate(tuple([a_tuple[i].x for i in range(len(a_tuple))]))
    rtn.y = np.concatenate(tuple([a_tuple[i].y for i in range(len(a_tuple))]))
    rtn.t = np.concatenate(tuple([a_tuple[i].t for i in range(len(a_tuple))]))
    rtn.ts = np.concatenate(tuple([a_tuple[i].ts for i in range(len(a_tuple))]))
    return rtn

    # np.concatenate(a_tuple)


# In[3]:

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
        
        
    
    
        
    

calculator = Calculator()
retina = RetinaRecord('blinking_2.data',calculator)
retina.packageStart = 1130
view = DataViewer(retina)


# In[ ]:



