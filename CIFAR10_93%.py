
# coding: utf-8

# # CNN : CIFAR-10
# 
# Dataset:<br>
# https://www.cs.toronto.edu/~kriz/cifar.html
# 
# 
# 2019-07-25
# 
# ### 学籍番号： 318000
# ### 氏　　名： 長浜 太郎

# ## 課題内容
# 
# 下記のモデル設計では、<font color="Red">過学習</font>の傾向がはっきりと現れてしまっているため、<br>
# 各自がモデルを改良して、精度と損失の２つのグラフで過学習していない状態であることを確認しつつ、より一層の精度の向上を目指す。
# 
# ### 参考：
# 　　https://keras.io/ja/getting-started/sequential-model-guide/

# ## 成績の判定基準：
# 
# <font color="Red">テスト用データ</font>に対する精度で成績を判定する。
# 
# - SS： 90 % 以上<br>
# - Ｓ： 88 % 以上 90 % 未満<br>
# - Ａ： 85 % 以上 88 % 未満<br>
# - Ｂ： 80 % 以上 85 % 未満<br>
# - Ｃ： 80 % 未満<br>
# - Ｄ： 出席をほとんどせず、課題も出さない場合
# 
# ## <font color="Red">過学習</font>の傾向がある場合は<font color="Red">提出しないこと！</font>
# 
# ## <font color="Red">提出期限</font>：
# ### 2019年07月28日(日) 18:00
# 
# - 提出先：<br>
# 　　k_wada@nagahama-i-bio.ac.jp<br>
# - 件名：<br>
# 　　プログラミング実習１の最終成果物<br>
# - 添付ファイル：<br>
# 　　本ファイルを編集したファイル (CIFAR10-Start.ipynb)<br>
# - 本文：<br>
# 　　学生番号と氏名

# ## 各種ライブラリのインポート

# In[1]:

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence

from keras import regularizers
from keras.initializers import he_normal

import keras.callbacks

import numpy as np
import datetime
import os

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

regularizer=regularizers.l2(1.0e-4)
initializer=he_normal()

# ## 計算開始時刻の記録

# In[2]:

start_time = datetime.datetime.now()
print(start_time)

# ## Numpy の乱数の種を設定
# 
# 再実行後も train, validation, test の各データセットが同一となるために必要

# In[3]:

seed = 456123789
np.random.seed(seed=seed)

# ## CIFAR10 のデータをロードし、学習に使用するデータセットの割合を設定

# In[4]:

# CIFAR10 のデータをロード
(X_train, Y_train), (x_test, y_test) = cifar10.load_data()

# データセットの最初の rate (0～1) の割合だけ使用
rate = 1.0
num_train = round(X_train.shape[0] * rate)
num_test  = round(x_test.shape[0]  * rate)

arange_train = np.arange(X_train.shape[0])
select_train = np.random.choice(arange_train, num_train, replace=False) # 重複なし

arange_test  = np.arange(x_test.shape[0])
select_test  = np.random.choice(arange_test , num_test , replace=False) # 重複なし

X_train = X_train[select_train]
Y_train = Y_train[select_train]

x_test  = x_test[select_test]
y_test  = y_test[select_test]

# ## CIFAR10 の正解ラベル名を設定

# In[5]:

labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ## 画像データを 0～1 の範囲に変換 (0-255 => 0-1)

# In[6]:

X_train = X_train.astype('float32')
x_test  = x_test.astype('float32')
X_train /= 255
x_test  /= 255

# ## 正解ラベルを One-hot 表現に変換

# In[7]:

Y_train = to_categorical(Y_train)
y_test  = to_categorical(y_test)

# ## 上記の train のデータを訓練用と検証用とに分割

# In[8]:

# 検証用データの割合
validation_rate = 0.2

# 訓練用データの個数
num_train = round(X_train.shape[0] * (1 - validation_rate))

x_train = X_train[:num_train]
x_valid = X_train[num_train:]
y_train = Y_train[:num_train]
y_valid = Y_train[num_train:]

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_valid.shape = ", x_valid.shape)
print("y_valid.shape = ", y_valid.shape)
print("x_test.shape  = ", x_test.shape)
print("y_test.shape  = ", y_test.shape)

# ## モデルの構築

# In[9]:

model = Sequential()

model.add(
    Conv2D(128, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer,
    input_shape=x_train.shape[1:]
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(
    Conv2D(128, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(256, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(
    Conv2D(256, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(512, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(
    Conv2D(512, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
          
model.add(
    Conv2D(512, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(
    Conv2D(512, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(1028, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(
    Conv2D(1028, (3, 3), 
    padding='same',
    kernel_regularizer=regularizer,
    kernel_initializer=initializer
    ))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,
    kernel_regularizer=regularizer,
    kernel_initializer=initializer))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

# ## モデルの概要を表示

# In[10]:

model.summary()

# ## モデルのコンパイル

# In[11]:

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])

# ## TensorBoard の設定
# 
# 下記のフォルダ内にある過去の学習ログのファイルを削除してから<br>
# コマンドプロンプト上で、この Notebook のファイルがある場所に移動した後で、<br>
# 以下のコマンドを実行する。<br>
# 
# `tensorboard --logdir="./logs"`
# 
# 上記のコマンドを実行後に、指示が表示されるので、ブラウザ上で<br>
# 
# `http://localhost:6006`
# 
# の URL を入力して TensorBoard を開く。
# 
# https://keras.io/ja/callbacks/#tensorboard

# In[12]:

log_dir = "logs"

if not (os.path.exists(log_dir) and os.path.isdir(log_dir)):
    os.mkdir(log_dir)
    
tb_cb = keras.callbacks.TensorBoard(
              log_dir=log_dir,
              write_graph=True,
              write_images=True)

def step_decay(epoch):
     initial_lrate = 0.001
     drop = 0.5
     epochs_drop = 10
     lrate = initial_lrate * np.power(drop, np.floor(epoch / epochs_drop))
     return lrate

lr_decay = keras.callbacks.LearningRateScheduler(step_decay)

# ## データの水増し

# In[13]:

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
)

train_datagen.fit(x_train)

batch_size = 128

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)


# ## 学習の実行

# In[14]:

aug_ratio = 10
steps_per_epoch  = (x_train.shape[0] // batch_size) * aug_ratio
epochs  = 150

history = model.fit_generator(
                    train_generator,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(x_valid, y_valid),
                    verbose=1,
                    callbacks=[lr_decay, tb_cb]
)

# ## 学習の履歴をグラフ表示

# In[15]:

# acc、val_acc のプロット
plt.figure()
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title("Training and validation accuracy")
plt.legend(loc="lower right")

# loss, val_loss のプロット
plt.figure()
plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training and validation loss")
plt.legend(loc="lower left")

plt.show()

# ## テスト用データを使った最終評価

# In[16]:

loss, score = model.evaluate(x_test, y_test, batch_size=32, verbose=0)

print('Test loss    :', loss)
print('Test accuracy:', score)

# ## テスト用データの先頭の 10 枚を可視化

# In[17]:

fig = plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.85, bottom=0.01)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    answer = np.argmax(y_test[i])
    plt.title(labels[answer])

plt.suptitle("The first ten of the test data", fontsize=20)
plt.show()

# ## テストデータの先頭の 10 枚の予測結果と正解ラベルを表示

# In[18]:

print("予測：")
pred = np.argmax(model.predict(x_test[:10]), axis=1)
s = ""
for i in range(10):
    s += "{:12s}".format(labels[pred[i]])
print(s)

print("\n正解：")
answer = np.argmax(y_test[:10], axis=1)
s = ""
for i in range(10):
    s += "{:12s}".format(labels[answer[i]])
print(s)

# ## 経過時間を表示

# In[19]:

end_time = datetime.datetime.now()
print("\nStart   Time  : " + str(start_time))
print(  "End     Time  : " + str(end_time))
print(  "Elapsed Time  : " + str(end_time - start_time))
