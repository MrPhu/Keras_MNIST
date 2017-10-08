#python D:\Python36\Git\Keras_MNIST\Recognition.py


import numpy as np
import keras.models
from keras.models import model_from_json
from skimage import io, color, transform
import matplotlib.pyplot as plt


path_to_project = "D:\\Python36\\Git\\Keras_MNIST\\" #change path_to_project depending on your project


#load jsonfile
path_to_json = path_to_project + "Trained_Model\\model.json"
json_file = open(path_to_json,'r')
loaded_model_json = json_file.read()
json_file.close()


#load weight
path_to_weights = path_to_project + "Trained_Model\\model.h5"
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(path_to_weights)

#re-compile model
loaded_model.compile(loss='categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])

#read created testset
path_to_testset = path_to_project + "TestSet\\7.png" #change test_set depending your test set
x = io.imread(path_to_testset)

#convert to grayscale
x_gray = color.rgb2gray(x)

#convert to black background
x_gray = 1. - x_gray

#show grayscale image
plt.imshow(x_gray,cmap=plt.cm.gray)
plt.show()

#resize and show model to 28x28
x_resize = transform.resize(x_gray,(28,28))
plt.imshow(x_resize,cmap=plt.cm.gray)
plt.show()

#reshape into matrix
x_input = x_resize.reshape(1,1,28,28).astype(np.float) 

#predict model
results = loaded_model.predict(x_input)#,batch_size=128, verbose=0) 
print('resultl matrix: ',results)
prediction = results.argmax(axis=1)
print('prediction: ',prediction)

