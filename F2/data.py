import torch
import numpy as np
from numpy.random import multivariate_normal
from random import shuffle, seed
import matplotlib.pyplot as plt
from torch.autograd import Variable

def getxy(x=0,y=0, num=1000,  std=0.5):
    return multivariate_normal(np.array([x,y]), 
                               np.array([[std,0],[0,std]]), num)

def pos():
    pos = []
    xy = getxy(0, 0.5, 100, 0.03)
    pos.append(xy)
    xy = getxy(0, 3.0, 100, 0.03)
    pos.append(xy)
    xy = getxy(2.25, 1.5, 800, 0.1)
    pos.append(xy)
    pos = np.concatenate(pos)
    return pos

def neg():
    neg = []
    xy = getxy(1.0, 0.75, 700, 0.03)
    neg.append(xy)
    xy = getxy(1.0, 1.5, 100, 0.03)
    neg.append(xy)
    xy = getxy(1.0, 2.5, 200, 0.03)
    neg.append(xy)
    neg = np.concatenate(neg)
    return neg

def plot(pos, neg):
    plt.plot(pos[:,0],pos[:,1], 'r^')
    plt.plot(neg[:,0],neg[:,1], 'bo')


def getXY(pos, neg):
    posy = np.ones((pos.shape[0],1))
    negy = -1 * np.ones((neg.shape[0],1))
    posxy = np.concatenate([pos,posy],1)
    negxy = np.concatenate([neg,negy],1)
    xy = np.concatenate([posxy,negxy],0)
    
    l = (list(range(len(xy))))
    seed(1)
    shuffle(l)
    xy = xy[l]
    XY = torch.from_numpy(xy)
    #XY = getXY(p,n)
    X = XY[:,:2].float()
    Y = XY[:,2].float()
    x = Variable(X)
    y = Variable(Y)
    return XY,x,y

def plotf(p,n,f):
    a = f.weight.data[0][0]
    b = f.weight.data[0][1]
    c = f.bias.data[0]
    a,b,c
    def getx(y):
        return -(b*y+c)/a
    y1,y2 = 0,4
    x1,x2 = getx(y1),getx(y2)
    plt.plot(p[:,0],p[:,1], 'r^')
    plt.plot(n[:,0],n[:,1], 'bo')
    plt.plot([x1,x2],[y1,y2], 'g')
        
    
