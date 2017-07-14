import torch
from loss import *


def train(model, criterion, x, y, epochs=20000, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  
    
    for epoch in range(epochs):
        optimizer.zero_grad()  
        outputs = model(x)
        loss = criterion(outputs,y)
        loss.backward()
        #model.weight.grad.data[0][0] = 0
        optimizer.step()
        
        if epoch%1000==0:  print(loss.data[0])


def trainl(model, criterion, x, y, epochs=20000, lr=0.01, a=0.4):
    f = model
    optimizer = torch.optim.SGD(list(model.parameters())+list(criterion.parameters()), lr=lr)  
    
    for epoch in range(epochs):
        optimizer.zero_grad()  
        outputs = model(x)
        loss = criterion(outputs,y)
        loss.backward()
        criterion.rev()
        
        optimizer.step()
        
        if epoch%100==0:  
            #print("grad", criterion.l.grad.data[0])
            #print(criterion.l.data[0],RatPg(f(x),y,a).data[0], RatPf(f(x),y).data[0])
            print(loss.data[0])



    
