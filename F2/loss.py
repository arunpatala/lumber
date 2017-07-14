import torch
from torch import nn
from torch.autograd import Variable

# zero one loss
def l01(fx,y):
     return ((y==1)*(fx<=0)) + ((y==-1)*(fx>0))
# true pos
def tp(fx,y):
    return ((y==1)*(1-l01(fx,y))).float().sum()
# false pos
def fp(fx,y):
    return ((y==-1)*(l01(fx,y))).float().sum()
# accuracy overall
def acc(fx,y):
    return (((y==1)*(fx>=0)) + ((y==-1)*(fx<0))).float().sum()/len(y)
# precision
def P(fx,y):
    return tp(fx,y)/(tp(fx,y)+fp(fx,y))
# recall
def R(fx,y):
    return tp(fx,y)/Yp(y)

#hinge loss
def lh(fx,y):
    return torch.clamp(1-y*fx,min=0)
def lhp(fx,y):
    return (y==1).float()*lh(fx,y)
def lhn(fx,y):
    return (y==-1).float()*lh(fx,y)
# true pos lower bound
def tpl(fx,y):
    return ((y==1).float()*(1-lh(fx,y))).sum()
# false pos upper bound
def fpu(fx,y):
    return ((y==-1).float()*(lh(fx,y))).sum()
# recall lower bound
def Rl(fx,y):
    return tpl(fx,y)/Yp(y)
# prec lower bound
def Pl(fx,y):
    return tpl(fx,y)/(tpl(fx,y)+fpu(fx,y))

# hinge loss on pos points
def Lp(fx,y):
    return ((y==1).float()*(lh(fx,y))).float().sum()
# hinge loss on neg points
def Ln(fx,y):
    return ((y==-1).float()*(lh(fx,y))).float().sum()
# num of pos
def Yp(y):
    return (y==1).float().sum()

# recall at prec condition
def RatPg(fx,y,a):
    return a * Ln(fx,y) + (1-a) * Lp(fx,y) - (1-a) * Yp(y)
# recall at prec min
def RatPf(fx,y):
    return Lp(fx,y)

# recall at prec 
def RatP(fx,y,l,a):
    return RatPf(fx,y) + l * RatPg(fx,y,a)

class RatPLoss(nn.Module):
    def __init__(self, a, l=0.1):
        super(RatPLoss, self).__init__()
        self.a = a
        self.l = nn.Parameter(torch.Tensor([l]))
        #self.l = Variable(torch.Tensor([l]))

    def forward(self, y_pred, y):
        return RatP(y_pred, y, self.l, self.a)/len(y)

    def rev(self):
        self.l.grad.data *= -1

# prec at recall condition
def PatRg(fx,y,b):
    return b - 1 + Lp(fx,y)/Yp(y)
# prec at recall min
def PatRf(fx,y):
    return Ln(fx,y)

# prec at recall  
def PatR(fx,y,l,b):
    return PatRf(fx,y) + l * PatRg(fx,y,b)

class PatRLoss(nn.Module):
    def __init__(self, a, l=0.1):
        super(PatRLoss, self).__init__()
        self.a = a
        self.l = nn.Parameter(torch.Tensor([l]))

    def forward(self, y_pred, y):
        return RatP(y_pred, y, self.l, self.a)/len(y)

# Area under the curve
class AUCPRLoss(nn.Module):
    def __init__(self, N=10, min=0.05, max=0.95):
        super(AUCPRLoss, self).__init__()
        
        self.a = torch.linspace(min, max, N)
        self.b = [nn.Parameter(torch.Tensor([0.1])) for a in self.a] 
        #print(self.a)
        self.raps = [RatPLoss(a) for a in self.a] 

    def forward(self, y_pred, y):
        loss = 0
        for r,b in zip(self.raps,self.b):
            bb = b[0].unsqueeze(0).expand(y_pred.size())
            loss += r(y_pred+bb,y)
        return loss/len(self.a)

    def rev(self):
        [r.rev() for r in self.raps]

    def parameters(self):
        for b in self.b:
            yield b
        for r in self.raps:
            for p in r.parameters():
                yield p

    def addb(self, i, y_pred):
        bb = self.b[i].unsqueeze(0).expand(y_pred.size())
        return y_pred+bb


class HingeLoss(nn.Module):
    def __init__(self, margin=0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y):
        return torch.clamp(1-y*(y_pred-self.margin),0).sum()/len(y)



