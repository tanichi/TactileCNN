# -*- coding: utf-8 -*-

import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers

import collections

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import sys

def opensound(file_path,length):
    # data : ここにwavデータがnumpy.ndarrayとして保持されます。
    # sampling_rate : 大半のwav音源のサンプリングレートは44.1kHzです
    # fmt : フォーマットはだいたいPCMでしょう
    data, sampling_rate = sf.read(file_path)

    NFFT = 1024 # フレームの大きさ
    OVERLAP = NFFT // 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
    frame_length = data.shape[0] # wavファイルの全フレーム数
    time_song = float(frame_length) / sampling_rate  # 波形長さ(秒)
    time_unit = 1 / float(sampling_rate) # 1サンプルの長さ(秒)
    #print("sampling rate", sampling_rate)

    # 💥 1.
    # FFTのフレームの時間を決めていきます
    # time_rulerに各フレームの中心時間が入っています
    start = (NFFT / 2) * time_unit
    stop = time_song
    step =  (NFFT - OVERLAP) * time_unit
    time_ruler = np.arange(start, stop, step)

    # 💥 2.
    # 窓関数は周波数解像度が高いハミング窓を用います
    window = np.hamming(NFFT)
    
    spec = np.zeros([len(time_ruler), 1 + int(NFFT / 2)]) #転置状態で定義初期化
    pos = 0

    for fft_index in range(len(time_ruler)):
        # 💥 1.フレームの切り出します
        frame = data[pos:pos+NFFT]
        # フレームが信号から切り出せない時はアウトです
        if len(frame) == NFFT:
            # 💥 2.窓関数をかけます
            windowed = window * frame
            # 💥 3.FFTして周波数成分を求めます
            # rfftだと非負の周波数のみが得られます
            fft_result = np.fft.rfft(windowed)
            # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
            # グラフで見やすくするために対数をとります
            fft_data = np.log(np.abs(fft_result) ** 2)
            # fft_data = np.log(np.abs(fft_result))
            # fft_data = np.abs(fft_result) ** 2
            # fft_data = np.abs(fft_result)
            # これで求められました。あとはspecに格納するだけです
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i-1] = fft_data[i]
                # 💥 4. 窓をずらして次のフレームへ
            pos += (NFFT - OVERLAP)

    ne=np.reshape(spec[:length*10],(length,10,513))

    ### プロットします
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
    
args = sys.argv
    
test_x = opensound(args[1],80)
                    
test_x = test_x-np.min(test_x)
test_x = test_x/np.max(test_x)
test_x = test_x.astype(np.float32)
test_x = test_x.reshape([test_x.shape[0],1,10,513])  # 必ず（データの総数, channel数, 縦, 横）の形にしておく
    
infer_net = CNN()
serializers.load_npz(
    'result/snapshot_iter_3372',
    infer_net, path='updater/model:main/predictor/')

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = infer_net(test_x).data
print(y)

c = collections.Counter(list(y.argmax(axis=1)))
print('予測ラベル:', y.argmax(axis=1))
print('予測頻度:',c.most_common())


'''
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
'''
