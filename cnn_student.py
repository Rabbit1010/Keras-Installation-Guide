#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:37:35 2018

@author: gaoyi
"""


''' python package'''
import numpy as np  # a powerful N-dimensional array package
from PIL import Image # help us to real the image & transform to array matrix
from glob import glob # glob the image path
from os.path import basename #split data name & label
import matplotlib.pyplot as plt # plot loss and feature map

''' keras package'''
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

''' set parameter '''
lr = 0.001
epoch = 16
batch_size = 32
adam = optimizers.Adam(lr=lr)
cls_list = ['Bread','Dairy product', 'Dessert', 'Egg', 
            'Fried food','Meat', 'Noodles/Pasta', 
            'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

def one_hot_encode(input,class_num):
    loc = np.arange(len(input))
    Y = np.zeros((len(input),class_num))
    Y[loc,input] = 1
    return Y

''' load preprocessed data '''
train_data = np.load("./training_28.npz")
x_train = train_data['x']
y_train = train_data['y']
y_train = one_hot_encode(y_train,len(cls_list))

eval_data = np.load('./validation_28.npz')
x_eval = eval_data['x']
y_eval = eval_data['y']
y_eval = one_hot_encode(y_eval,len(cls_list))

#data_gen = ImageDataGenerator(rescale=1. / 255)
#
#train_generator = data_gen.flow_from_directory('/home/gaoyi/workshop/food 11/training_class',
#                                               target_size=(64, 64),
#                                               batch_size=batch_size,
#                                               class_mode='categorical')
#
#validation_generator = data_gen.flow_from_directory('/home/gaoyi/workshop/food 11/validation_class',
#                                               target_size=(64, 64),
#                                               batch_size=batch_size,
#                                               class_mode='categorical')

''' set model '''

def cnn_model(alpha = 0.):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28 ,3),
                            activation='relu'
                            ,name='conv1',kernel_regularizer=regularizers.l2(alpha))) 
    # input 32,28,28,3 output 32,26,26,32
    
    model.add(Conv2D(32, (3, 3),
                            activation='relu'
                            ,name='conv2',kernel_regularizer=regularizers.l2(alpha)))
    # input 32,26,26,3 output 32,24,24,32
    
    model.add(Conv2D(64, (3, 3),
                            activation='relu'
                            ,name='conv3',kernel_regularizer=regularizers.l2(alpha)))
    # input 32,24,24,3 output 32,22,22,32
    
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool'))
    
    model.add(Flatten())
    
    model.add(Dense(512,activation='relu',
                    name='dense1',kernel_regularizer=regularizers.l2(alpha)))
    model.add(Dense(256,activation='relu',
                    name='dense2',kernel_regularizer=regularizers.l2(alpha))) #512 ->256
    model.add(Dense(11,activation='softmax',
                    name='dense3',kernel_regularizer=regularizers.l2(alpha))) #256 ->11
    
    return model

'''start to train'''

print('Bulid model...')
model = cnn_model(alpha=0.0)
model.compile(loss="mean_squared_error",
              optimizer="sgd",
              metrics=['accuracy'])

print('Training...')
hist = model.fit(x_train, y_train,
                 batch_size = batch_size,
                 epochs = epoch,
                 validation_data = (x_eval,y_eval))


### learning curve, accuracy rate of training and validation sets
plt.figure()
plt.plot(hist.history['loss'],'#4D80E6',label='loss',linewidth=1.8)
#plt.plot(hist.history['val_loss'],'#FF8033',label='val_loss',linewidth=1.8)
plt.title('Learning curve ')
plt.ylabel('Cross-entropy')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.figure()
plt.plot(hist.history['acc'],'#4D80E6',label='acc',linewidth=1.8)
plt.plot(hist.history['val_acc'],'#FF8033',label='val_acc',linewidth=1.8)
plt.title('Training accuracy ')
plt.ylabel('Accuracy rate')
plt.xlabel('Epoch')
plt.legend(loc='best')
loss, acc = model.evaluate( x_eval,y_eval, batch_size=batch_size)
print('Evaluaion set accuracy %.3f %%'%(acc*100))

### plot distribution of weights and biases
weights = model.get_weights()
for i in [0,2,4,6,8,10]:
    plt.figure()
    plt.hist(weights[i].flatten(),bins=80)
    if i <= 4:
        plt.title('Histogram of conv%d' %(i//2+1))
    else:
        plt.title('Histogram of dense%d' %(i//2-2))
    plt.ylabel('Number')
    plt.xlabel('Value')


### show detected and undetected food images
pred = model.predict(x_eval,batch_size=batch_size)
y_pred = np.argmax(pred,axis = 1)
y_true = np.argmax(y_eval,axis = 1)
right_idx = [ i for i,v in enumerate(y_pred) if y_pred[i] == y_true[i]]
wrong_idx = [ i for i,v in enumerate(y_pred) if y_pred[i] != y_true[i]]
w = wrong_idx[0]
true = y_true[w]
pred = y_pred[w]
plt.figure()
plt.imshow(x_eval[w])
plt.xlabel('pred: %s,    label:%s' %(cls_list[pred],cls_list[true]))

r = right_idx[0]
true = y_true[r]
pred = y_pred[r]
plt.figure()
plt.imshow(x_eval[r])
plt.xlabel('pred: %s,    label:%s' %(cls_list[pred],cls_list[true]))

### show feature map
inp = model.input
outputs = [layer.output for layer in model.layers]
functor = K.function([inp]+ [K.learning_phase()], outputs )
test = x_eval[:5].reshape(-1,28,28,3)
layer_outs = functor([test, 1.])   
plt.figure()
plt.imshow(x_eval[1])
for i in range(5):
    plt.figure()
    plt.imshow(layer_outs[1][i,:,:,0])

#%%
### L2 regularization
print('Bulid model...')
model_L2 = cnn_model(alpha = 1e-4)
model_L2.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print('Training...')
hist_L2 = model_L2.fit(x_train, y_train,
                 batch_size = batch_size,
                 epochs = epoch,
                 validation_data = (x_eval,y_eval))
### learning curve, accuracy rate of training and validation sets
plt.figure()
plt.plot(hist_L2.history['loss'],'#4D80E6',label='loss',linewidth=1.8)
plt.title('L2 Learning curve ')
plt.ylabel('Cross-entropy')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.figure()
plt.plot(hist_L2.history['acc'],'#4D80E6',label='acc',linewidth=1.8)
plt.plot(hist_L2.history['val_acc'],'#FF8033',label='val_acc',linewidth=1.8)
plt.title('L2 Training accuracy ')
plt.ylabel('Accuracy rate')
plt.xlabel('Epoch')
plt.legend(loc='best')


### plot distribution of weights and biases
L2_weights = model_L2.get_weights()
for i in [0,2,4,6,8,10]:
    plt.figure()
    plt.hist(L2_weights[i].flatten(),bins=80)
    if i <= 4:
        plt.title('L2 Histogram of conv%d' %(i//2+1))
    else:
        plt.title('L2 Histogram of dense%d' %(i//2-2))
    plt.ylabel('Number')
    plt.xlabel('Value')