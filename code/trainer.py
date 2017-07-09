from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchsample.modules import ModuleTrainer
from amazonds import AmazonDataset, CLASS_NAMES, BinaryAccuracy
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from torch.autograd import Variable
from torchsample.callbacks import ModelCheckpoint


batch_size = 16
epochs = 30
num_classes = len(CLASS_NAMES)

model = resnet50(pretrained=True)
model.fc = nn.Sequential( nn.Linear(model.fc.in_features, num_classes), nn.Sigmoid())

trainer = ModuleTrainer(model.cuda())

trainer.compile(loss=nn.BCELoss().cuda(), 
    optimizer='adam', 
    metrics=[BinaryAccuracy()],
    callbacks=[ModelCheckpoint(directory = "../input/torch/", 
                filename='torch{epoch}.{loss}.pth.tar', monitor='val_loss')])

from torchsample import TensorDataset
from torch.utils.data import DataLoader

train_dataset = AmazonDataset("train35.csv")
x,y = train_dataset[0]

train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = AmazonDataset("valid5.csv")
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#help(trainer.fit_loader)
trainer.fit_loader(train_loader, val_loader=val_loader, nb_epoch=epochs, cuda_device=0)


