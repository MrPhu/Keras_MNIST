#python D:\Python36\Git\Keras_MNIST\KR_CNN2.py
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy

from keras.datasets import mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

#batchsize 128
#numclass 10
#epoch 12

#reshape dataset (depth, row, column)
X_train = X_train.reshape(X_train.shape[0],1,28,28) # "image_data_format" : "channels_first",
X_test = X_test.reshape(X_test.shape[0],1,28,28)

#normalize dataset
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#convert class vector to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#create model L
def create_model():
    #add layers
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape = (1,28,28)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation = 'softmax'))
    #compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    #Adadelta() because adaptive learning method, simple
    #cross entropy because we have multiple classes
    #fit model
    model.fit(X_train,y_train,batch_size = 128,epochs = 12,verbose=1,validation_data=(X_test,y_test))
    return model

#evaluate model
model = create_model()
scores = model.evaluate(X_test,y_test,verbose=0)
print('Loss: ',scores[0])
print('Acuracy: ',scores[1])

#save model
#serialize model to JSON
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
