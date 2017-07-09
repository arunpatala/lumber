import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

import cv2
from tqdm import tqdm, trange

from sklearn.metrics import fbeta_score

# Params

input_size = 128
input_channels = 3

epochs = 10
batch_size = 16
learning_rate = 0.0001
lr_decay = 1e-4

valid_data_size = 5000  # Samples to withhold for validation

model = Sequential()
model.add(BatchNormalization(input_shape=(input_channels, input_size, input_size)))
model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))
opt = Adam(lr=learning_rate, decay=lr_decay)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])

from keras.models import load_model

model = load_model('../input/weights.11-0.9640.h5')

vdatagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    fill_mode="reflect")


df_train_data = pd.read_csv('../input/train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))
print(labels)
labels = sorted(labels)
print(labels)
label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
print(inv_label_map)
x_valid = []
y_valid = []

df_valid = df_train_data[(len(df_train_data) - valid_data_size):]
print(df_valid.values[:10])
for f, tags in tqdm(df_valid.values, miniters=100):
    img = cv2.resize(cv2.imread('../input/train-jpg/{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_valid.append(img)
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)
x_valid = np.array(x_valid, np.float32)

val_gen = vdatagen.flow(x_valid.transpose(0,3,1,2), batch_size=batch_size)
#p_valid = model.predict(x_valid.transpose(0,3,1,2), batch_size=batch_size)

print(model.metrics_names)
print(model.evaluate(x_valid.transpose(0,3,1,2), y_valid, batch_size=batch_size))

p_valid = model.predict(x_valid.transpose(0,3,1,2))
print("valid shape",p_valid.shape)

def get_cutoffs():
    cutoffs = 0.2 * np.ones(len(labels))
    max_score = fbeta_score(y_valid, np.array(p_valid) > cutoffs, beta=2, average='samples')
    #print(0.20, max_score)
    besti = 0.2
    for i in np.arange(0.05,0.95,0.025):
        cutoffs = i * np.ones(len(labels))
        score = fbeta_score(y_valid, np.array(p_valid) > cutoffs, beta=2, average='samples')
        if score > max_score:
            max_score = score
            besti = i
        #print(i, score, max_score)


    cutoffs = besti * np.ones(len(labels))
    print(cutoffs, max_score)
    for i in range(len(labels)):
        ci = cutoffs[i]
        for j in np.arange(0.05,0.95,0.025):
            cutoffs[i] = j
            score = fbeta_score(y_valid, np.array(p_valid) > cutoffs, beta=2, average='samples')
            if score > max_score: 
                max_score = score 
                ci = j
        cutoffs[i] = ci
        #print(i,max_score,cutoffs)
    print(cutoffs,max_score)
    return cutoffs

cutoffs = get_cutoffs()
print(fbeta_score(y_valid, np.array(p_valid) > cutoffs, beta=2, average='samples'), cutoffs)


val_gen = vdatagen.flow(x_valid.transpose(0,3,1,2), y_valid, batch_size=batch_size)
x_train = []
y_train = []

df_train = df_train_data[:(len(df_train_data) - valid_data_size)]

for f, tags in tqdm(df_train.values, miniters=500):
    try:
        img = cv2.resize(cv2.imread('../input/train-jpg/{}.jpg'.format(f)), (input_size, input_size))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train .append(img)
        y_train.append(targets)
    except: print(f)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)

train_gen = datagen.flow(x_train.transpose(0,3,1,2), y_train, batch_size=batch_size)

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint('../input/weights.{epoch:02d}-{val_acc:.4f}.h5')]

opt = Adam(lr=learning_rate, decay=lr_decay)


model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])
#model = load_model('../input/weights.01-0.96.h5')
"""model.fit(x_train.transpose(0,3,1,2),
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(x_valid.transpose(0,3,1,2), y_valid))
"""
model.fit_generator(train_gen, len(x_train)//batch_size, epochs=epochs, callbacks=callbacks,        
    validation_data=val_gen, validation_steps=len(x_valid)//batch_size)




p_valid = model.predict(x_valid.transpose(0,3,1,2), batch_size=batch_size)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

cutoffs = get_cutoffs()
print(fbeta_score(y_valid, np.array(p_valid) > cutoffs, beta=2, average='samples'), cutoffs)


df_test_data = pd.read_csv('../input/sample_submission_v2.csv')


l = 5000
y_test = []
for i in trange(0,len(df_test_data.values),l):
    x_test = []
    for f, tags in df_test_data.values[i:i+l]:
        img = cv2.resize(cv2.imread('../input/test-jpg/{}.jpg'.format(f)), (input_size, input_size))
        x_test.append(img)
    x_test = np.array(x_test, np.float32)
    #print("x_test",x_test.shape)
    p_test = model.predict(x_test.transpose(0,3,1,2), batch_size=batch_size)
    #print("p_test",p_test.shape)
    y_test.append(p_test)
    #if len(y_test)==3: break






print("y0", y_test[0].shape)
result = np.concatenate(y_test,axis=0)
print("result", result.shape)
result = pd.DataFrame(result, columns=labels)

preds = []

for i in tqdm(range(result.shape[0]), miniters=500):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > cutoffs, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)

# 0.918
