
# coding: utf-8

# # CNN : CIFAR-10
# 
# Dataset:<br>
# https://www.cs.toronto.edu/~kriz/cifar.html
# 
# 
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

# In[2]:

start_time = datetime.datetime.now()
print(start_time)

# In[3]:

seed = 456123789
np.random.seed(seed=seed)

# In[4]:
(X_train, Y_train), (x_test, y_test) = cifar10.load_data()


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



# In[5]:

labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# In[6]:

X_train = X_train.astype('float32')
x_test  = x_test.astype('float32')
X_train /= 255
x_test  /= 255


# In[7]:

Y_train = to_categorical(Y_train)
y_test  = to_categorical(y_test)


# In[8]:

validation_rate = 0.2

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


# In[10]:

model.summary()


# In[11]:

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])

# `tensorboard --logdir="./logs"`
# 
# `http://localhost:6006`
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

# In[15]:

plt.figure()
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title("Training and validation accuracy")
plt.legend(loc="lower right")

plt.figure()
plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training and validation loss")
plt.legend(loc="lower left")

plt.show()

# In[16]:

loss, score = model.evaluate(x_test, y_test, batch_size=32, verbose=0)

print('Test loss    :', loss)
print('Test accuracy:', score)


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


# In[18]:

print("prediction：")
pred = np.argmax(model.predict(x_test[:10]), axis=1)
s = ""
for i in range(10):
    s += "{:12s}".format(labels[pred[i]])
print(s)

print("\nanswer：")
answer = np.argmax(y_test[:10], axis=1)
s = ""
for i in range(10):
    s += "{:12s}".format(labels[answer[i]])
print(s)

# In[19]:

end_time = datetime.datetime.now()
print("\nStart   Time  : " + str(start_time))
print(  "End     Time  : " + str(end_time))
print(  "Elapsed Time  : " + str(end_time - start_time))
