---

layout: post
title: "DL for Audio Classification"
author: "Xhy"
categories: Speech
tags: [improve]
image: blake-connally-ipXPK5F7hao.jpg
---

Photo by Blake Connally

### This is a study note about using deep learning to classify audio. I pretty much appreciate [Seth Adams's](https://www.youtube.com/user/seth8141/playlists) effort. Thank You very much :)

<br />


## Table of Contents

* [Plotting & Cleaning][1]
* [Model Preparation (CNN / RNN)][2]
* [Saving Data and Models][3]
* [Predictions][4]

[1]:	#1
[2]:	#2
[3]: #3
[4]: #4



<br />

<h3 id="1"> 1. Plotting & Cleaning</h3>

```python
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(13,4))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(13,4))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(13,4))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(13,4))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# Clean the noise floor
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return(Y, freq)
    
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate


classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distrubution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[0, 0]
#    print(wav_file)
    signal, rate = librosa.load('wavfiles/'+wav_file, sr=44100)
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(signal)
    
    mask = envelope(signal, rate, 0.0005 )
    signal = signal[mask]
#    plt.figure()
#    plt.subplot(212)
#    plt.plot(signal)
    
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()
```

![png](/assets/img/DLAC/output_2_0.png)



![png](/assets/img/DLAC/output_2_1.png)



![png](/assets/img/DLAC/output_2_2.png)



![png](/assets/img/DLAC/output_2_3.png)



![png](/assets/img/DLAC/output_2_4.png)


### Downsample & Saving wav


```python
if len(os.listdir('clean_jupyter/')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/' + f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean_jupyter/'+f, rate=rate, data=signal[mask])
```

    100%|████████████████████████████████████████████████████████████████████████████████| 300/300 [01:02<00:00,  4.80it/s]

<h3 id="2"> 2. Model Preparation (CNN / RNN)</h3>

```python
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

def build_rand_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean_jupyter/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=10)
    return X, y        
  
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

 
def get_recurrent_model():
    #shape of data RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model     
 
class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)  
        
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean_jupyter/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()    
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
    
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat), 
                                    y_flat)
model.fit(X, y, epochs=20, batch_size=1024,
          shuffle=True,
          class_weight=class_weight)
```

    Using TensorFlow backend.



![png](/assets/img/DLAC/output_6_1.png)


    100%|███████████████████████████████████████████████████████████████████████████| 26410/26410 [00:57<00:00, 458.27it/s]


    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 13, 9, 16)         160       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 13, 9, 32)         4640      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 13, 9, 64)         18496     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 9, 128)        73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 6, 4, 128)         0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 6, 4, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3072)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               393344    
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 499,402
    Trainable params: 499,402
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/20
    26410/26410 [==============================] - ETA: 56s - loss: 2.2970 - acc: 0.13 - ETA: 27s - loss: 2.2831 - acc: 0.13 - ETA: 18s - loss: 2.2682 - acc: 0.14 - ETA: 13s - loss: 2.2567 - acc: 0.15 - ETA: 10s - loss: 2.2552 - acc: 0.15 - ETA: 8s - loss: 2.2492 - acc: 0.1533 - ETA: 7s - loss: 2.2415 - acc: 0.156 - ETA: 6s - loss: 2.2367 - acc: 0.156 - ETA: 5s - loss: 2.2318 - acc: 0.156 - ETA: 4s - loss: 2.2265 - acc: 0.157 - ETA: 4s - loss: 2.2242 - acc: 0.159 - ETA: 3s - loss: 2.2215 - acc: 0.159 - ETA: 3s - loss: 2.2166 - acc: 0.160 - ETA: 2s - loss: 2.2121 - acc: 0.162 - ETA: 2s - loss: 2.2087 - acc: 0.164 - ETA: 2s - loss: 2.2031 - acc: 0.170 - ETA: 1s - loss: 2.1967 - acc: 0.172 - ETA: 1s - loss: 2.1904 - acc: 0.175 - ETA: 1s - loss: 2.1833 - acc: 0.178 - ETA: 1s - loss: 2.1747 - acc: 0.182 - ETA: 0s - loss: 2.1659 - acc: 0.187 - ETA: 0s - loss: 2.1541 - acc: 0.193 - ETA: 0s - loss: 2.1435 - acc: 0.198 - ETA: 0s - loss: 2.1319 - acc: 0.202 - ETA: 0s - loss: 2.1203 - acc: 0.209 - 5s 172us/step - loss: 2.1091 - acc: 0.2141
    Epoch 20/20
    26410/26410 [==============================] - ETA: 1s - loss: 0.1849 - acc: 0.934 - ETA: 1s - loss: 0.2230 - acc: 0.926 - ETA: 1s - loss: 0.2142 - acc: 0.927 - ETA: 1s - loss: 0.2098 - acc: 0.928 - ETA: 1s - loss: 0.2071 - acc: 0.927 - ETA: 1s - loss: 0.2083 - acc: 0.928 - ETA: 1s - loss: 0.2111 - acc: 0.927 - ETA: 1s - loss: 0.2093 - acc: 0.927 - ETA: 1s - loss: 0.2106 - acc: 0.927 - ETA: 1s - loss: 0.2150 - acc: 0.925 - ETA: 1s - loss: 0.2179 - acc: 0.925 - ETA: 0s - loss: 0.2155 - acc: 0.926 - ETA: 0s - loss: 0.2116 - acc: 0.927 - ETA: 0s - loss: 0.2107 - acc: 0.927 - ETA: 0s - loss: 0.2115 - acc: 0.927 - ETA: 0s - loss: 0.2135 - acc: 0.927 - ETA: 0s - loss: 0.2120 - acc: 0.927 - ETA: 0s - loss: 0.2138 - acc: 0.926 - ETA: 0s - loss: 0.2156 - acc: 0.925 - ETA: 0s - loss: 0.2168 - acc: 0.924 - ETA: 0s - loss: 0.2164 - acc: 0.925 - ETA: 0s - loss: 0.2190 - acc: 0.924 - ETA: 0s - loss: 0.2190 - acc: 0.924 - ETA: 0s - loss: 0.2214 - acc: 0.923 - ETA: 0s - loss: 0.2232 - acc: 0.922 - 2s 70us/step - loss: 0.2244 - acc: 0.9217
<h3 id="3"> 3. Saving Data and Models</h3>

### cfg_jupyter.py


```python
import os
class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)  
        self.model_path = os.path.join('models_jupyter', mode + '.model')
        self.p_path = os.path.join('pickles_jupyter', mode + '.p')
```


```python
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg_jupyter import Config
 
def check_data():
    if os.path.isfile(config.p_path):
        print('Loaing existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
    
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean_jupyter/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=10)
    config.data = (X, y)
    
    #save the train data
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    return X, y        
  
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
 
def get_recurrent_model():
    #shape of data RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
     
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean_jupyter/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


config = Config(mode='time')

if config.mode == 'conv':    #epochs=20 acc: 0.938
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()    
elif config.mode == 'time': #epochs=20 acc: 0.76070 epochs=200 acc: 0.975
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
    
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat), 
                                    y_flat)

checkpoint = ModelCheckpoint(config.model_path,
                             monitor='val_acc', 
                             verbose=1, 
                             mode='max',
                             save_best_only=True,
                             save_weights_only=False,
                             period=1)
model.fit(X, y, epochs=20, batch_size=1024,
          shuffle=True, validation_split=0.1,
          callbacks=[checkpoint])
    
model.save(config.model_path)
```

![png](/assets/img/DLAC/output_10_0.png)

    100%|███████████████████████████████████████████████████████████████████████████| 26410/26410 [00:59<00:00, 443.51it/s]


    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 9, 128)            72704     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 9, 128)            131584    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 9, 128)            0         
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 9, 64)             8256      
    _________________________________________________________________
    time_distributed_2 (TimeDist (None, 9, 32)             2080      
    _________________________________________________________________
    time_distributed_3 (TimeDist (None, 9, 16)             528       
    _________________________________________________________________
    time_distributed_4 (TimeDist (None, 9, 8)              136       
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 72)                0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 10)                730       
    =================================================================
    Total params: 216,018
    Trainable params: 216,018
    Non-trainable params: 0
    _________________________________________________________________
    Train on 23769 samples, validate on 2641 samples
    Epoch 1/20
    23769/23769 [==============================] - ETA: 16s - loss: 2.3002 - acc: 0.08 - ETA: 8s - loss: 2.2924 - acc: 0.1084 - ETA: 5s - loss: 2.2876 - acc: 0.116 - ETA: 4s - loss: 2.2831 - acc: 0.120 - ETA: 3s - loss: 2.2764 - acc: 0.128 - ETA: 2s - loss: 2.2725 - acc: 0.129 - ETA: 2s - loss: 2.2691 - acc: 0.126 - ETA: 2s - loss: 2.2660 - acc: 0.128 - ETA: 1s - loss: 2.2627 - acc: 0.127 - ETA: 1s - loss: 2.2595 - acc: 0.127 - ETA: 1s - loss: 2.2565 - acc: 0.127 - ETA: 1s - loss: 2.2551 - acc: 0.127 - ETA: 1s - loss: 2.2511 - acc: 0.129 - ETA: 0s - loss: 2.2487 - acc: 0.129 - ETA: 0s - loss: 2.2461 - acc: 0.131 - ETA: 0s - loss: 2.2421 - acc: 0.133 - ETA: 0s - loss: 2.2394 - acc: 0.135 - ETA: 0s - loss: 2.2366 - acc: 0.137 - ETA: 0s - loss: 2.2323 - acc: 0.140 - ETA: 0s - loss: 2.2282 - acc: 0.142 - ETA: 0s - loss: 2.2233 - acc: 0.143 - ETA: 0s - loss: 2.2189 - acc: 0.145 - ETA: 0s - loss: 2.2143 - acc: 0.146 - 2s 93us/step - loss: 2.2137 - acc: 0.1470 - val_loss: 2.1058 - val_acc: 0.1912
    
    Epoch 00001: val_acc improved from -inf to 0.19122, saving model to models_jupyter\time.model
    
    Epoch 20/20
    23769/23769 [==============================] - ETA: 1s - loss: 0.7692 - acc: 0.744 - ETA: 1s - loss: 0.7506 - acc: 0.746 - ETA: 1s - loss: 0.7537 - acc: 0.739 - ETA: 1s - loss: 0.7591 - acc: 0.735 - ETA: 1s - loss: 0.7609 - acc: 0.735 - ETA: 0s - loss: 0.7637 - acc: 0.733 - ETA: 0s - loss: 0.7657 - acc: 0.731 - ETA: 0s - loss: 0.7654 - acc: 0.732 - ETA: 0s - loss: 0.7640 - acc: 0.734 - ETA: 0s - loss: 0.7611 - acc: 0.736 - ETA: 0s - loss: 0.7511 - acc: 0.740 - ETA: 0s - loss: 0.7524 - acc: 0.739 - ETA: 0s - loss: 0.7532 - acc: 0.739 - ETA: 0s - loss: 0.7537 - acc: 0.739 - ETA: 0s - loss: 0.7527 - acc: 0.739 - ETA: 0s - loss: 0.7487 - acc: 0.740 - ETA: 0s - loss: 0.7482 - acc: 0.740 - ETA: 0s - loss: 0.7460 - acc: 0.741 - ETA: 0s - loss: 0.7537 - acc: 0.738 - ETA: 0s - loss: 0.7500 - acc: 0.739 - ETA: 0s - loss: 0.7512 - acc: 0.740 - ETA: 0s - loss: 0.7541 - acc: 0.739 - ETA: 0s - loss: 0.7543 - acc: 0.739 - 1s 59us/step - loss: 0.7538 - acc: 0.7395 - val_loss: 0.6872 - val_acc: 0.7607
    
    Epoch 00020: val_acc improved from 0.75502 to 0.76070, saving model to models_jupyter\time.model
<h3 id="4"> 4. Predictions</h3>


```python
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle
from cfg_jupyter import Config

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()      
    return y_true, y_pred, fn_prob                              
    
df = pd.read_csv('instruments.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles_jupyter', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
print(config.model_path)
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean_jupyter')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
print(acc_score)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
    
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions.csv', index=False)       
```

    models_jupyter\conv.model
    Extracting features from audio


    100%|████████████████████████████████████████████████████████████████████████████████| 300/300 [00:52<00:00,  5.71it/s]


    0.920990114184995

