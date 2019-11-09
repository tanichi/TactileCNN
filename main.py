# -*- coding: utf-8 -*-

import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import sys

def opensound(file_path,length):
    data, sampling_rate = sf.read(file_path)

    NFFT = 1024 # フレームの大きさ
    OVERLAP = NFFT // 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
    frame_length = data.shape[0] # wavファイルの全フレーム数
    time_song = float(frame_length) / sampling_rate  # 波形長さ(秒)
    time_unit = 1 / float(sampling_rate) # 1サンプルの長さ(秒)
    #print("sampling rate", sampling_rate)

    #  1.
    # FFTのフレームの時間を決めていきます
    # time_rulerに各フレームの中心時間が入っています
    start = (NFFT / 2) * time_unit
    stop = time_song
    step =  (NFFT - OVERLAP) * time_unit
    time_ruler = np.arange(start, stop, step)

    #  2.
    # 窓関数は周波数解像度が高いハミング窓を用います
    window = np.hamming(NFFT)
    
    spec = np.zeros([len(time_ruler), 1 + int(NFFT / 2)]) #転置状態で定義初期化
    pos = 0

    for fft_index in range(len(time_ruler)):
        #  1.フレーム切り出し
        frame = data[pos:pos+NFFT]
        # 長さをみる
        if len(frame) == NFFT:
            #  2.窓関数をかける
            windowed = window * frame
            #  3.FFTして周波数成分を求める
            # rfftだと非負の周波数のみ
            fft_result = np.fft.rfft(windowed)
            #  4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗
            # グラフで見やすくするために対数をとる
            fft_data = np.log(np.abs(fft_result) ** 2)
            # fft_data = np.log(np.abs(fft_result))
            # fft_data = np.abs(fft_result) ** 2
            # fft_data = np.abs(fft_result)
            # あとはspecに格納するだけです
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i-1] = fft_data[i]
                #  4. 窓をずらして次のフレームへ
            pos += (NFFT - OVERLAP)

    ne=np.reshape(spec[:length*10],(length,10,513))

    ### プロット
    # matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
    '''
    plt.imshow(ne[2].T, extent=[0, time_song, 0, sampling_rate/2], aspect="auto")
    #plt.imshow(spec.T, extent=[0, time_song, 0, sampling_rate/2], aspect="auto")
    plt.xlabel("time[s]")
    plt.ylabel("frequency[Hz]")
    plt.colorbar()
    plt.show()
    '''
    return ne

class CNN(Chain):  
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(1, 20, 3)
            self.cn2 = L.Convolution2D(20, 50, 3)
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(500, 4)

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        h3 = F.dropout(F.relu(self.fc1(h2)))
        return self.fc2(h3)
    
class AlexNet(Chain):
    def __init__(self, num_class=3, train=True):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 96, 11, stride=2)
            self.conv2=L.Convolution2D(None, 256, 5, pad=2)
            self.conv3=L.Convolution2D(None, 384, 3, pad=1)
            self.conv4=L.Convolution2D(None, 384, 3, pad=1)
            self.conv5=L.Convolution2D(None, 256, 3, pad=1)
            self.fc6=L.Linear(None, 4096)
            self.fc7=L.Linear(None, 4096)
            self.fc8=L.Linear(None, num_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h

# 各音声ファイルを開く
train_x = np.vstack((opensound("01-60s.wav",562),opensound("02-60s.wav",562)))
train_x = np.vstack((train_x,opensound("03-60s.wav",562)))
train_x = np.vstack((train_x,opensound("paper-60s.wav",562)))

#print(np.max(new),np.min(new),np.average(new))
# np.float 0.0-1.0に 正規化
train_x = train_x-np.min(train_x)
train_x = train_x/np.max(train_x)
train_x = train_x.astype(np.float32)

train_y = np.vstack((np.full((562,1),0),np.full((562,1),1)))
train_y = np.vstack((train_y,np.full((562,1),2)))
train_y = np.vstack((train_y,np.full((562,1),3)))
train_y = train_y.astype(np.int32)

test_x = np.vstack((opensound("01-10s.wav",93), opensound("02-10s.wav",93)))
test_x = np.vstack((test_x, opensound("03-10s.wav",93)))
test_x = np.vstack((test_x, opensound("paper-10s.wav",93)))

test_x = test_x-np.min(test_x)
test_x = test_x/np.max(test_x)
test_x = test_x.astype(np.float32)

test_y = np.vstack((np.full((93,1),0),np.full((93,1),1)))
test_y = np.vstack((test_y,np.full((93,1),2)))
test_y = np.vstack((test_y,np.full((93,1),3)))
test_y = test_y.astype(np.int32)

train_x = train_x.reshape([2248,1,10,513])  # 必ず（データの総数, channel数, 縦, 横）の形にしておく
test_x = test_x.reshape([372,1,10,513])  # 必ず（データの総数, channel数, 縦, 横）の形にしておく
train_y = train_y.reshape(2248)
test_y = test_y.reshape(372)
train = tuple_dataset.TupleDataset(train_x, train_y)
test = tuple_dataset.TupleDataset(test_x, test_y)
sys.exit()
model = L.Classifier(CNN())
optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = iterators.SerialIterator(train, batch_size=100)
test_iter = iterators.SerialIterator(test, batch_size=20, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (150, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
