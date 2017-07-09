from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchsample.metrics import Metric

DATA_DIR ='../input/'

CLASS_NAMES=[
    'clear',    	 # 0
    'haze',	         # 1
    'partly_cloudy', # 2
    'cloudy',	     # 3
    'primary',	     # 4
    'agriculture',   # 5
    'water',	     # 6
    'cultivation',	 # 7
    'habitation',	 # 8
    'road',	         # 9
    'slash_burn',	 # 10
    'conventional_mine', # 11
    'bare_ground',	     # 12
    'artisinal_mine',	 # 13
    'blooming',	         # 14
    'selective_logging', # 15
    'blow_down',	     # 16
]

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def acc_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


## custom data loader -----------------------------------
class AmazonDataset(Dataset):

    def __init__(self, csv ='train35.csv', size=224, transform=transform, ds="train"):
        class_names = CLASS_NAMES
        num_classes = len(class_names)
        
        df_train = pd.read_csv(DATA_DIR + os.sep + "split" + os.sep + csv)
        num = len(df_train)
        label_map = {l: i for i, l in enumerate(class_names)}
        
        x,y = [],[]
        for f, tags in tqdm(df_train.values, miniters=100):
            targets = np.zeros(num_classes)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            x.append(f)
            y.append(targets)
        y = torch.from_numpy(np.array(y))

        #save
        self.transform = transform
        self.num       = num
        self.y         = y
        self.x         = x
        self.class_names = class_names
        self.df     = df_train
        self.f = DATA_DIR + os.sep + "{}-jpg".format(ds) + os.sep + "{}.jpg"


    def images(self, index):
        return loader(self.f.format(index))
        
    def __getitem__(self, index):
        img = self.images(self.x[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[index]
        return img.float(), label.float()


    def __len__(self):
        return self.num

def split(df_train_data, seed=0, valid_data_size=5000):
    df_train_data = df_train_data.sample(frac=1,random_state=seed).reset_index(drop=True)
    cut = (len(df_train_data) - valid_data_size)
    df_valid = df_train_data[cut:]
    df_train = df_train_data[:cut]
    return df_train, df_valid
    




def trainval():
    df_train_data = pd.read_csv('../input/train.csv')
    df_train, df_valid = split(df_train_data)
    df_valid.to_csv('../input/split/valid5.csv', index=False)
    df_train.to_csv('../input/split/train35.csv', index=False)


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true.long()).float().sum().data[0]
        self.total_count += (y_pred.numel())
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy



if __name__ == '__main__':
    ds = AmazonDataset('train35.csv')
    x,y = (ds[0])
    print(x.min(), x.max())
    print(y)


