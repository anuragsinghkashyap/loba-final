# loba-final
import tensorflow as tf
tf.__version__

import os
import itertools
from sklearn.datasets import load_files
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from tqdm import tqdm
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow import lite
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D
from keras.layers.core import Dense, Dropout

from keras.callbacks import ModelCheckpoint

from jupyterthemes import jtplot
jtplot.style(theme='grade3', context='notebook', ticks=True, grid=False)

train_dataset = load_files(os.getcwd() + r'/Dataset/Train/', shuffle=False)
test_dataset = load_files(os.getcwd() + r'/Dataset/Test/', shuffle=False)
validation_dataset = load_files(os.getcwd() + r'/Dataset/Validation/', shuffle=False)

train_files = train_dataset['filenames']
train_targets = train_dataset['target']

test_files = test_dataset['filenames']
test_targets = test_dataset['target']

validation_files = validation_dataset['filenames']
validation_targets = validation_dataset['target']

print('\nNumber of videos in training data:', train_files.shape[0])
print('Number of videos in test data:', test_files.shape[0])
print('Number of videos in validation data:', validation_files.shape[0])

for pair in zip(train_files[:5], train_targets[:5]):
    print(pair)
    
class Videos(object):
    def __init__ (self, target_size = (128,128), max_frames = 30):
        self.target_size = target_size
        self.max_frames = max_frames
        
    def read_videos(self, paths):
        list_of_videos = [self._read_video(path) for path in tqdm(paths)]
        tensor = np.vstack(list_of_videos)
        min_ = np.min(tensor, axis=(1,2,3), keepdims=True)
        max_ = np.max(tensor, axis=(1,2,3), keepdims=True)
        return ((tensor.astype('float32') - min_) / (max_ - min_))
    
    def _read_video(self, path):
        list_of_frames = []
        cap = cv2.VideoCapture(path)
        
        while True:
            ret,frame = cap.read()
            
            if (ret == False) or (len(list_of_frames) == self.max_frames):
                cap.release()
                break
                
            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = img_to_array(frame)
            list_of_frames.append(frame)
            
        temp_video = np.stack(list_of_frames)
        return np.expand_dims(temp_video, axis = 0)
    
# An object of the class `Videos` to load the data in the required format
reader = Videos(target_size=(128,128), max_frames=30)


# Reading training videos and one-hot encoding the training labels
X_train = reader.read_videos(train_files)
y_train = to_categorical(train_targets, num_classes=4)
print('Shape of training data:', X_train.shape)
print('Shape of training labels:', y_train.shape)


# Reading training videos and one-hot encoding the training labels
X_valid = reader.read_videos(validation_files)
y_valid = to_categorical(validation_targets, num_classes=4)
print('Shape of validation data:', X_valid.shape)
print('Shape of validation labels:', y_valid.shape)

# After pre-processing
# Displaying the first frame of the first processed video from the training data
plt.imshow(np.squeeze(X_train[0][0], axis=2), cmap='gray')

model = Sequential()

# Adding Alternate convolutional and pooling layers
model.add(Conv3D(filters=16, kernel_size=(5, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', 
                 input_shape=X_train.shape[1:]))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=64, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=256, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(Conv3D(filters=1024, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu'))
model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

# A global average pooling layer to get a 1-d vector
# The vector will have a depth (same as number of elements in the vector) of 1024
model.add(GlobalAveragePooling3D())

# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(4, activation='softmax'))

model.summary()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Saving the model that performed the best on the validation set
checkpoint = ModelCheckpoint(filepath='Model_2_weights_best.h5', 
                             save_best_only=True, verbose=1)

# Training the model for 40 epochs
history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_valid, y_valid), callbacks=[checkpoint])

# Loading the model that performed the best on the validation set
model.load_weights('Model_2_weights_best.h5')
# Testing the model on the Test data
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))

# Making the plot larger
plt.figure(figsize=(12, 8))

loss = history.history['loss']                   # Loss on the training data
val_loss = history.history['val_loss']          # Loss on the validation data
epochs = range(1, 51)

plt.plot(epochs, loss, 'ro-', label='Training Loss')
plt.plot(epochs, val_loss, 'go-', label = 'Validation Loss')
plt.legend()

def confusion_matrix_plot(cm, classes, 
                          title='Normalized Confusion Matrix', 
                          normalize=False, 
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        
    plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
 def confusion_matrix_plot(cm, classes, 
                          title='Normalized Confusion Matrix', 
                          normalize=False, 
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        
    plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    arr = np.array(y_test)
Y_test = np.array([np.where(r==1)[0][0] for r in arr])
confusion_matrix(Y_test, predictions)
confusion_matrix_plot = confusion_matrix_plot(cm, 
                                              classes=['Basketball', 'Cricket', 'Table Tennis', 'Volleyball'], 
                                              normalize=True)
       
