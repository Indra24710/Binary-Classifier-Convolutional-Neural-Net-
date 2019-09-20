//importing keras
from tensorflow import keras;

//importing Sequential to build a Sequential model
from keras.models import Sequential;

//importing the convolutional neural network class
from keras.layers import Conv2D;

//importing Dense for the dense hidden layer
from keras.layers import Dense;

//importing MaxPooling and flatten to reduce the size of the image and convert the 2d matrix or tensor to a single continuos vector
from keras.layers import MaxPooling2D;
from keras.layers import Flatten;

import numpy as np
//the imagedatagenerator class which is used to preprocess the input images 
from keras.preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator;

//setting the classifier to sequential type
classifier=Sequential();
//adding the convolutional layer
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'));
//performing maxpooling to the image and then flattening it 
classifier.add(MaxPooling2D(pool_size=(2,2)));
classifier.add(Flatten());

//creating the dense hidden layer
classifier.add(Dense(units=128,activation='relu'));
//final output neuron which uses sigmoid function as activation to classify the image into either of the two classes
classifier.add(Dense(units=1,activation='sigmoid'));
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']);  

//preprocessing the training and test data set before feeding it to the network
traindatagen= ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True);
testdatagen=ImageDataGenerator(rescale=1./255);
training_set=traindatagen.flow_from_directory('training_set',target_size=(64,64),batch_size=32,class_mode='binary');
test_set=testdatagen.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='binary');
classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=15,validation_data=test_set,validation_steps=2000);

//testing our trained model
test_image = image.load_img('images.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
     prediction = 'object1'
else:
     prediction = 'object2'
