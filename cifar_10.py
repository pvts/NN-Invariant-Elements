# -*- coding: utf-8 -*-

#Importing the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate, warp, AffineTransform
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, \
                                    Flatten, GlobalMaxPooling2D, \
                                    GlobalAveragePooling2D, AveragePooling2D
np.random.seed(42)
tf.random.set_seed(42)

#setting the directory path
Dir = '/content/drive/My Drive/models_cifar'

"""## Preprocessing"""

#label names
cifar_labels = ["airplane",  #index 0
                "automobile ", # index 1
                "bird",    # index 2 
                "cat",     # index 3 
                "deer",    # index 4
                "dog",     # index 5
                "frog",    # index 6 
                "horse",   # index 7 
                "ship",    # index 8 
                "truck"]   # index 9

#Load the dataset 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalizing the dataset to values between 0-1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print('Original Image Shape:', X_test[0].shape)

#padding zeros to expand the background size to 56x56
X_train = np.pad(X_train, ((0,0), (12,12), (12,12), (0,0)), mode='constant')
X_test = np.pad(X_test, ((0,0), (12,12), (12,12), (0,0)), mode='constant')

print('New Image Shape:', X_test[0].shape)

#converting classes to 1s and 0s
no_classes = 10
y_train_cat = to_categorical(y_train, no_classes)
y_test_cat = to_categorical(y_test, no_classes)

#setting the input
inp = X_train[0].shape

#showing some images with added background
figure = plt.figure(figsize=(5, 7))
for i, index in enumerate(np.random.choice(X_test.shape[0], size=12,
                                           replace=False)):
  ax = figure.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(X_train[index]))
  true_index = np.argmax(y_train_cat[index])
  ax.set_title("{}".format(cifar_labels[true_index]), color=("black"))
  plt.savefig('images.png')

"""## Image Transformations"""

#function to change the magnitude of shear
def shear(magnitude, test_set):
  afine_tf = AffineTransform(shear=magnitude)
  test_shear = np.zeros(test_set.shape)
  for index, y in enumerate(test_set):
    test_shear[index] = warp(y, inverse_map=afine_tf)
  return test_shear

test_first_shear = shear(0.2, X_test)
test_second_shear = shear(0.35, X_test)
test_third_shear = shear(0.5, X_test)

#function to change the magnitude of rotation
def rotation(magnitude, test_set):
  test_rotation = np.zeros(test_set.shape)
  for index, y in enumerate(test_set):
     test_rotation[index] = rotate(y, magnitude)
  return test_rotation

test_first_rotation = rotation(20, X_test)
test_second_rotation = rotation(40, X_test)
test_third_rotation = rotation(60, X_test)
test_fourth_rotation = rotation(90, X_test)

#function to set translation levels
def translation(dim, test_set, train_set):
  translation_matrix = np.float32(dim)
  width, height = train_set[0].shape[:2]
  test_translation = np.zeros(test_set.shape)
  for index, y in enumerate(test_set):
    test_translation[index] = cv2.warpAffine(y, translation_matrix, 
                                             (width, height))
  return test_translation
    
test_first_translation = translation([[1,0,1], [0,1,1]], X_test, X_train)
test_second_translation = translation([ [1,0,2], [0,1,2] ], X_test, X_train)
test_third_translation = translation([ [1,0,4], [0,1,4] ], X_test, X_train)
test_fourth_translation = translation([ [1,0,8], [0,1,8] ], X_test, X_train)
test_fifth_translation = translation([ [1,0,14], [0,1,14] ], X_test, X_train)

#Vertically flipping all the images in the test set
X_test_vflipped = np.zeros_like(X_test)
for index, i in enumerate(X_test):
    X_test_vflipped[index] = np.fliplr(i)

#Horizontally flipping all the images in the test set
X_test_hflipped = np.zeros_like(X_test)
for index, i in enumerate(X_test):
    X_test_hflipped[index] = np.flipud(i)

"""## Data Augmentation"""

opt = optimizers.Adam(lr=0.001)
#Data augmentation using the MLP model
def build_mlp_model():
  model = Sequential()
  model.add(Dense(32, activation = 'relu', input_shape=inp))
  model.add(Dense(64, activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(256, activation = 'relu'))
  #output layer
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#CNN model
def build_cnn_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#Initializing the DataGen, specifying one transformation at a time
datagen = ImageDataGenerator(rotation_range=90, validation_split=0.1)

datagen.fit(X_train)

train_generator = datagen.flow(X_train, y_train_cat, 
                               batch_size=32, subset='training')

validation_generator = datagen.flow(X_train, y_train_cat, batch_size=32, 
                                    subset='validation')

#mlp
model = build_mlp_model()
history = model.fit(train_generator, validation_data=validation_generator,
                    steps_per_epoch=len(train_generator)/32,
                    validation_steps=len(validation_generator)/32,
                    epochs=300, workers=6)

model.save_weights('aug_mlp_rot.h5')

#mlp accuracy
score_mlp_frotation_aug = model.evaluate(test_first_rotation, y_test_cat)
score_mlp_srotation_aug = model.evaluate(test_second_rotation, y_test_cat)
score_mlp_trotation_aug = model.evaluate(test_third_rotation, y_test_cat)
score_mlp_ftrotation_aug = model.evaluate(test_fourth_rotation, y_test_cat)

#cnn rotation on augmented data
model_2 = build_cnn_model()

history_2 = model_2.fit_generator(train_generator, validation_data=validation_generator, 
                                  steps_per_epoch=len(train_generator)/32,
                                  validation_steps=len(validation_generator)/32,
                                  epochs=300, workers=6)

#model_2.save_weights('aug_cnn_rot.h5')

#cnn rotation results
score_cnn_frotation_aug = model_2.evaluate(test_first_rotation, y_test_cat)
score_cnn_srotation_aug = model_2.evaluate(test_second_rotation, y_test_cat)
score_cnn_trotation_aug = model_2.evaluate(test_third_rotation, y_test_cat)
score_cnn_ftrotation_aug = model_2.evaluate(test_fourth_rotation, y_test_cat)

"""Translation"""

datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, 
                             validation_split=0.1)
#Initializing the DataGen, specifying one transformation at a time
datagen.fit(X_train)

train_generator = datagen.flow(X_train, y_train_cat, 
                               batch_size=32, subset='training')

validation_generator = datagen.flow(X_train, y_train_cat, batch_size=32, 
                                    subset='validation')

history_3 = model.fit_generator(generator=train_generator,
                                validation_data=validation_generator,
                                steps_per_epoch=len(train_generator)/32,
                                validation_steps=len(validation_generator)/32,
                                epochs=300,
                                workers=6)

model.save_weights('aug_mlp_tr.h5')

score_mlp_first_tr_aug = model.evaluate(test_first_translation, y_test_cat)
score_mlp_second_tr_aug = model.evaluate(test_second_translation, y_test_cat)
score_mlp_third_tr_aug = model.evaluate(test_third_translation, y_test_cat)
score_mlp_fourth_tr_aug = model.evaluate(test_fourth_translation, y_test_cat)
score_mlp_fifth_tr_aug = model.evaluate(test_fifth_translation, y_test_cat)

#cnn on augmented data (translation)
history_4 = model_2.fit_generator(generator=train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=len(train_generator)/32,
                                  validation_steps=len(validation_generator)/32,
                                  epochs=300,
                                  workers=6)
model_2.save_weights('aug_cnn_tr.h5')

#cnn data augmentation translation scores
score_cnn_first_tr_aug = model_2.evaluate(test_first_translation, y_test_cat)
score_cnn_second_tr_aug = model_2.evaluate(test_second_translation, y_test_cat)
score_cnn_third_tr_aug = model_2.evaluate(test_third_translation, y_test_cat)
score_cnn_fourth_tr_aug = model_2.evaluate(test_fourth_translation, y_test_cat)
score_cnn_fifth_tr_aug = model_2.evaluate(test_fifth_translation, y_test_cat)

"""## Models"""

#MaxPooling CNN model
def build_cnn_mp_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#CNN + AP model
def build_cnn_ap_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(AveragePooling2D(2, 2))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(AveragePooling2D(2, 2))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(AveragePooling2D(2, 2))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#CNN + Dropout model
def build_cnn_do_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(Dropout(0.3))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Dropout(0.3))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#GlobalMaxPooling CNN model
def build_cnn_globalmp_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(GlobalMaxPooling2D())
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#Global Average Pooling CNN model
def build_cnn_globalap_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=inp))
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(GlobalAveragePooling2D())
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))
  model.compile(optimizer=opt, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

#MLP Model
checkpoint = ModelCheckpoint(filepath='mlp.model.hdf5', verbose=1, 
                             save_best_only=True)
model_mlp = build_mlp_model()
history_mlp = model_mlp.fit(X_train, y_train_cat, epochs=10, batch_size=128, 
                            verbose=0, validation_split=0.1, 
                            callbacks=checkpoint)

#CNN Model
checkpoint = ModelCheckpoint(filepath='cnn.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn = build_cnn_model()
history_cnn = model_cnn.fit(X_train, y_train_cat, epochs=10, batch_size=128, 
                            verbose=1, validation_split=0.1, 
                            callbacks=checkpoint)

#CNN + MaxPooling Model
checkpoint = ModelCheckpoint(filepath='cnn_mp.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn_mp = build_cnn_mp_model()
history_cnn_mp = model_cnn_mp.fit(X_train, y_train_cat, epochs=10, 
                                  batch_size=128, verbose=1, 
                                  validation_split=0.1, callbacks=checkpoint)

#CNN + AveragePooling Model
checkpoint = ModelCheckpoint(filepath='cnn_ap.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn_ap = build_cnn_ap_model()
history_cnn_ap = model_cnn_ap.fit(X_train, y_train_cat, epochs=10, 
                                  batch_size=128, verbose=1, 
                                  validation_split=0.1, callbacks=checkpoint)

#CNN + Dropout Model
checkpoint = ModelCheckpoint(filepath='cnn_do.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn_do = build_cnn_do_model()
history_cnn_do = model_cnn_do.fit(X_train, y_train_cat, epochs=10, 
                                  batch_size=128, verbose=1, 
                                  validation_split=0.3, callbacks=checkpoint)

#CNN + GlobalMaxPooling
checkpoint = ModelCheckpoint(filepath='cnn_globalmp.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn_global_mp = build_cnn_globalmp_model()
history_global_mp = model_cnn_global_mp.fit(X_train, y_train_cat, epochs=10, 
                                  batch_size=128, verbose=1, 
                                  validation_split=0.1, callbacks=checkpoint)

#CNN + GlobalAveragePooling 
checkpoint = ModelCheckpoint(filepath='cnn_globalap.model.hdf5', verbose=1, 
                             save_best_only=True)

model_cnn_global_ap = build_cnn_globalap_model()
history_global_ap = model_cnn_global_ap.fit(X_train, y_train_cat, epochs=10, 
                                  batch_size=128, verbose=1, 
                                  validation_split=0.1, callbacks=checkpoint)

"""## Results"""

#Mlp Model - Load best weights
model_mlp.load_weights(filepath='mlp.model.hdf5')

score_mlp = model_mlp.evaluate(X_test, y_test_cat, verbose=0)

#Small Translated Mlp Model
score_first_tr_mlp = model_mlp.evaluate(test_first_translation, y_test_cat, 
                                         verbose=0)

#Small Translated Mlp Model
score_second_tr_mlp = model_mlp.evaluate(test_second_translation, y_test_cat, 
                                         verbose=0)

#Medium Translated Mlp Model
score_third_tr_mlp = model_mlp.evaluate(test_third_translation, y_test_cat, 
                                          verbose=0)

#Medium Translated Mlp Model
score_fourth_tr_mlp = model_mlp.evaluate(test_fourth_translation, y_test_cat, 
                                          verbose=0)

#Large Translated MLP
score_fifth_tr_mlp = model_mlp.evaluate(test_fifth_translation, y_test_cat, 
                                        verbose=0)

#First Rotation MLP
score_frotation_mlp = model_mlp.evaluate(test_first_rotation, y_test_cat, 
                                         verbose=0)

#Second Rotation Mlp Model
score_srotation_mlp = model_mlp.evaluate(test_second_rotation, y_test_cat, 
                                         verbose=0)

#Third Rotation Mlp Model
score_trotation_mlp = model_mlp.evaluate(test_third_rotation, y_test_cat, 
                                         verbose=0)

#Fourth Rotation Mlp Model
score_ftrotation_mlp = model_mlp.evaluate(test_fourth_rotation, y_test_cat, 
                                         verbose=0)

#Vertically Flipped Model
score_vflp_mlp = model_mlp.evaluate(X_test_vflipped, y_test_cat, verbose=0)

#Horizontally Flipped Model
score_hflp_mlp = model_mlp.evaluate(X_test_hflipped, y_test_cat, verbose=0)

#Shear MLP Model
score_fshear_mlp = model_mlp.evaluate(test_first_shear, y_test_cat, verbose=0)

#Shear MLP Model
score_sshear_mlp = model_mlp.evaluate(test_second_shear, y_test_cat, verbose=0)

#Shear MLP Model
score_tshear_mlp = model_mlp.evaluate(test_third_shear, y_test_cat, verbose=0)

"""---


**CNN**

---
"""

#Load CNN Model
model_cnn.load_weights(filepath=Dir+'/cnn.model.hdf5')

score_cnn = model_cnn.evaluate(X_test, y_test_cat, verbose=0)

#Small Translated CNN Model
score_first_tr_cnn = model_cnn.evaluate(test_first_translation, y_test_cat, 
                                         verbose=0)

#Small Translated CNN Model
score_second_tr_cnn = model_cnn.evaluate(test_second_translation, y_test_cat, 
                                         verbose=0)

#Medium Translated CNN Model
score_third_tr_cnn = model_cnn.evaluate(test_third_translation, y_test_cat, 
                                        verbose=0)

#Medium Translated CNN Model
score_fourth_tr_cnn = model_cnn.evaluate(test_fourth_translation, y_test_cat, 
                                         verbose=0)

#Large Translated CNN Model
score_fifth_tr_cnn = model_cnn.evaluate(test_fifth_translation, y_test_cat, 
                                        verbose=0)

#First Rotation CNN
score_frotation_cnn = model_cnn.evaluate(test_first_rotation, y_test_cat, 
                                         verbose=0)

#Second Rotation CNN
score_srotation_cnn = model_cnn.evaluate(test_second_rotation, y_test_cat, 
                                         verbose=0)

#Third Rotation CNN
score_trotation_cnn = model_cnn.evaluate(test_third_rotation, y_test_cat, 
                                         verbose=0)

#Fourth Rotation CNN Model
score_ftrotation_cnn = model_cnn.evaluate(test_fourth_rotation, y_test_cat, 
                                         verbose=0)

#Vertically Flipped CNN Model
score_vflp_cnn = model_cnn.evaluate(X_test_vflipped, y_test_cat, verbose=0)

#Horizontally Flipped CNN Model
score_hflp_cnn = model_cnn.evaluate(X_test_hflipped, y_test_cat, verbose=0)

#Shear CNN Model
score_fshear_cnn = model_cnn.evaluate(test_first_shear, y_test_cat, verbose=0)

#Shear CNN Model
score_sshear_cnn = model_cnn.evaluate(test_second_shear, y_test_cat, verbose=0)

#Shear CNN Model
score_tshear_cnn = model_cnn.evaluate(test_third_shear, y_test_cat, verbose=0)

"""---


> **CNN + MaxPooling**



---
"""

#CNN + MaxPooling
model_cnn_mp.load_weights(Dir+'/cnn_mp.model.hdf5')

score_cnn_mp = model_cnn_mp.evaluate(X_test, y_test_cat, verbose=0)

#Small Translated CNN + MaxPooling
score_first_tr_cnnmp = model_cnn_mp.evaluate(test_first_translation, y_test_cat, 
                                              verbose=0)

#Small Translated CNN +MaxPooling
score_second_tr_cnnmp = model_cnn_mp.evaluate(test_second_translation, y_test_cat, 
                                              verbose=0)

#Medium Translated CNN + MaxPooling
score_third_tr_cnnmp = model_cnn_mp.evaluate(test_third_translation, 
                                             y_test_cat, verbose=0)

#Medium Translated CNN + MaxPooling
score_fourth_tr_cnnmp = model_cnn_mp.evaluate(test_fourth_translation, 
                                              y_test_cat, verbose=0)

#Bigger Translated CNN MP Model
score_fifth_tr_cnnmp = model_cnn_mp.evaluate(test_fifth_translation, y_test_cat, 
                                             verbose=0)

#First Rotation CNN MP Model
score_frotation_cnnmp = model_cnn_mp.evaluate(test_first_rotation, y_test_cat, 
                                              verbose=0)

#Second Rotation CNN MP Model
score_srotation_cnnmp = model_cnn_mp.evaluate(test_second_rotation, y_test_cat, 
                                              verbose=0)

#Third Rotation CNN MP Model
score_trotation_cnnmp = model_cnn_mp.evaluate(test_third_rotation, y_test_cat, 
                                              verbose=0)

#Fourth Rotation CNN MP Model
score_ftrotation_cnnmp = model_cnn_mp.evaluate(test_fourth_rotation, y_test_cat, 
                                               verbose=0)

#Vertically Flipped MaxPooling Model
score_vflp_mp = model_cnn_mp.evaluate(X_test_vflipped, y_test_cat, verbose=0)

#Horizontally Flipped MaxPooling Model
score_hflp_mp = model_cnn_mp.evaluate(X_test_hflipped, y_test_cat, verbose=0)

#Shear CNN MP Model
score_fshear_mp = model_cnn_mp.evaluate(test_first_shear, y_test_cat, verbose=0)

#Shear CNN MP Model 
score_sshear_mp = model_cnn_mp.evaluate(test_second_shear, y_test_cat, verbose=0)

#Shear Model 
score_tshear_mp = model_cnn_mp.evaluate(test_third_shear, y_test_cat, verbose=0)

"""---


> **CNN + AveragePooling**



---
"""

#CNN + AveragePooling
model_cnn_ap.load_weights(Dir+'/cnn_ap.model.hdf5')

score_cnn_ap = model_cnn_ap.evaluate(X_test, y_test_cat, verbose=0)

#Small Translated CNN + AveragePooling
score_first_tr_cnnap = model_cnn_ap.evaluate(test_first_translation, 
                                             y_test_cat, verbose=0)

#Small Translated CNN + AveragePooling
score_second_tr_cnnap = model_cnn_ap.evaluate(test_second_translation, 
                                              y_test_cat, verbose=0)

#Medium Translated CNN + AveragePooling
score_third_tr_cnnap = model_cnn_ap.evaluate(test_third_translation, 
                                             y_test_cat, verbose=0)

#Medium Translated CNN + AveragePooling
score_fourth_tr_cnnap = model_cnn_ap.evaluate(test_fourth_translation, 
                                              y_test_cat, verbose=0)

#Bigger Translated CNN AveragePooling
score_fifth_tr_cnnap = model_cnn_ap.evaluate(test_fifth_translation, y_test_cat, 
                                             verbose=0)

#First Rotation CNN AveragePooling
score_frotation_cnnap = model_cnn_ap.evaluate(test_first_rotation, y_test_cat, 
                                         verbose=0)

#Second Rotation CNN AveragePooling
score_srotation_cnnap = model_cnn_ap.evaluate(test_second_rotation, y_test_cat, 
                                         verbose=0)

#Third Rotation CNN AveragePooling
score_trotation_cnnap = model_cnn_ap.evaluate(test_third_rotation, y_test_cat, 
                                         verbose=0)

#Fourth Rotation CNN AveragePooling Model
score_ftrotation_cnnap = model_cnn_ap.evaluate(test_fourth_rotation, y_test_cat, 
                                            verbose=0)

#Vertically Flipped AveragePooling Model
score_vflp_ap = model_cnn_ap.evaluate(X_test_vflipped, y_test_cat, verbose=0)

#Horizontally Flipped AveragePooling Model
score_hflp_ap = model_cnn_ap.evaluate(X_test_hflipped, y_test_cat, verbose=0)

#Shear Model
score_fshear_ap = model_cnn_ap.evaluate(test_first_shear, y_test_cat, verbose=0)

#Shear Model
score_sshear_ap = model_cnn_ap.evaluate(test_second_shear, y_test_cat, verbose=0)

#Shear Model
score_tshear_ap = model_cnn_ap.evaluate(test_third_shear, y_test_cat, verbose=0)

"""---


> **CNN + Dropout**



---
"""

#CNN + Dropout
model_cnn_do.load_weights(filepath=Dir+'/cnn_do.model.hdf5')

score_cnn_do = model_cnn_do.evaluate(X_test, y_test_cat, verbose=0)

#Small Translation CNN + Dropout
score_first_tr_do = model_cnn_do.evaluate(test_first_translation, y_test_cat,
                                          verbose=0)

#Small Translation CNN + Dropout
score_second_tr_do = model_cnn_do.evaluate(test_second_translation, y_test_cat, 
                                           verbose=0)

#Medium Translation CNN + Dropout
score_third_tr_do = model_cnn_do.evaluate(test_third_translation, y_test_cat,
                                          verbose=0)

#Medium Translation CNN + Dropout
score_fourth_tr_do = model_cnn_do.evaluate(test_fourth_translation, y_test_cat,
                                           verbose=0)

#Large Translation CNN + Dropout
score_fifth_tr_do = model_cnn_do.evaluate(test_fifth_translation, y_test_cat, 
                                          verbose=0)

#First Rotation CNN + Dropout
score_frotation_do = model_cnn_do.evaluate(test_first_rotation, y_test_cat,
                                           verbose=0)

#Second Rotation CNN + Dropout
score_srotation_do = model_cnn_do.evaluate(test_second_rotation, y_test_cat, 
                                           verbose=0)

#Third Rotation CNN + Dropout
score_trotation_do = model_cnn_do.evaluate(test_third_rotation, y_test_cat, 
                                           verbose=0)

#Fourth Rotation CNN DO Model
score_ftrotation_do = model_cnn_do.evaluate(test_fourth_rotation, y_test_cat, 
                                            verbose=0)

#Vertically Flipped CNN + Dropout Model
score_vflp_do = model_cnn_do.evaluate(X_test_vflipped, y_test_cat, 
                                      verbose=0)

#Horizontally Flipped CNN + Dropout Model
score_hflp_do = model_cnn_do.evaluate(X_test_hflipped, y_test_cat, 
                                      verbose=0)

#Shear Model
score_fshear_do = model_cnn_do.evaluate(test_first_shear, y_test_cat, verbose=0)

#Shear Model
score_sshear_do = model_cnn_do.evaluate(test_second_shear, y_test_cat, verbose=0)

#Shear Model
score_tshear_do = model_cnn_do.evaluate(test_third_shear, y_test_cat, verbose=0)

"""---


> **CNN + GlobalMaxPooling**



---
"""

#CNN Global MaxPooling Model
model_cnn_global_mp.load_weights(filepath=Dir+'/cnn_globalmp.model.hdf5')

#CNN Global MaxPooling
score_global_mp = model_cnn_global_mp.evaluate(X_test, y_test_cat, verbose=0)

#Small Translation
score_first_tr_global_mp = model_cnn_global_mp.evaluate(test_first_translation, 
                                                        y_test_cat, verbose=0)

#Small Translation
score_second_tr_global_mp = model_cnn_global_mp.evaluate(test_second_translation, 
                                                         y_test_cat, verbose=0)

#Medium Translation
score_third_tr_global_mp = model_cnn_global_mp.evaluate(test_third_translation, 
                                                        y_test_cat, verbose=0)

#Medium Translation
score_fourth_tr_global_mp = model_cnn_global_mp.evaluate(test_fourth_translation, 
                                                         y_test_cat, verbose=0)

#Large Translation
score_fifth_tr_global_mp = model_cnn_global_mp.evaluate(test_fifth_translation, 
                                                        y_test_cat, verbose=0)

#First Rotation CNN + GlobalMaxPooling
score_frotation_global_mp = model_cnn_global_mp.evaluate(test_first_rotation, 
                                                         y_test_cat, verbose=0)

#Second Rotation CNN GlobalMaxPooling
score_srotation_global_mp = model_cnn_global_mp.evaluate(test_second_rotation, 
                                                         y_test_cat, verbose=0)

#Third Rotation CNN GlobalMaxPooling
score_trotation_global_mp = model_cnn_global_mp.evaluate(test_third_rotation, 
                                                         y_test_cat, verbose=0)

#Fourth Rotation CNN GlobalMaxPooling
score_ftrotation_global_mp = model_cnn_global_mp.evaluate(test_fourth_rotation, 
                                                          y_test_cat, verbose=0)

#Horizontally Flipped
score_hflp_global_mp = model_cnn_global_mp.evaluate(X_test_hflipped, y_test_cat,
                                                    verbose=0)

#Vertically Flipped
score_vflp_global_mp = model_cnn_global_mp.evaluate(X_test_vflipped, y_test_cat,
                                                    verbose=0)

#Shear Model
score_fshear_globalmp = model_cnn_global_mp.evaluate(test_first_shear, 
                                                     y_test_cat, verbose=0)

#Shear Model
score_sshear_globalmp = model_cnn_global_mp.evaluate(test_second_shear, 
                                                     y_test_cat, verbose=0)

#Shear Model
score_tshear_globalmp = model_cnn_global_mp.evaluate(test_third_shear, 
                                                     y_test_cat, verbose=0)

"""---


> **CNN + GlobalAveragePooling**



---
"""

#Global AveragePooling Model
model_cnn_global_ap.load_weights(filepath=Dir+'/cnn_globalap.model.hdf5')

score_global_ap = model_cnn_global_ap.evaluate(X_test, y_test_cat, verbose=0)

#Translation Global AVGPooling Model
score_first_tr_global_ap = model_cnn_global_ap.evaluate(test_first_translation, 
                                                        y_test_cat, verbose=0)

#Translation Global AVGPooling Model
score_second_tr_global_ap = model_cnn_global_ap.evaluate(test_second_translation, 
                                                         y_test_cat, verbose=0)

#Translation Global AVGPooling Model
score_third_tr_global_ap = model_cnn_global_ap.evaluate(test_third_translation, 
                                                        y_test_cat, verbose=0)

#Translation Global AVGPooling Model
score_fourth_tr_global_ap = model_cnn_global_ap.evaluate(test_fourth_translation, 
                                                         y_test_cat, verbose=0)

#Translation Global AVGPooling Model
score_fifth_tr_global_ap = model_cnn_global_ap.evaluate(test_fifth_translation, 
                                                        y_test_cat, verbose=0)

#First Rotation CNN + GlobalAveragePooling
score_frotation_global_ap = model_cnn_global_ap.evaluate(test_first_rotation, 
                                                         y_test_cat, verbose=0)

#Second Rotation CNN + GlobalAveragePooling
score_srotation_global_ap = model_cnn_global_ap.evaluate(test_second_rotation,
                                                         y_test_cat, verbose=0)

#Third Rotation CNN + GlobalAveragePooling
score_trotation_global_ap = model_cnn_global_ap.evaluate(test_third_rotation, 
                                                         y_test_cat, verbose=0)

#Fourth Rotation CNN GlobalMaxPooling
score_ftrotation_global_ap = model_cnn_global_ap.evaluate(test_fourth_rotation, 
                                                          y_test_cat, verbose=0)

#Horizontally Flipped
score_hflp_global_ap = model_cnn_global_ap.evaluate(X_test_hflipped, y_test_cat,
                                                    verbose=0)

#Vertically Flipped
score_vflp_global_ap = model_cnn_global_ap.evaluate(X_test_vflipped, y_test_cat,
                                                    verbose=0)

#Shear Model
score_fshear_globalap = model_cnn_global_ap.evaluate(test_first_shear, 
                                                     y_test_cat, verbose=0)

#Shear Model
score_sshear_globalap = model_cnn_global_ap.evaluate(test_second_shear, 
                                                     y_test_cat, verbose=0)

#Shear Model
score_tshear_globalap = model_cnn_global_ap.evaluate(test_third_shear, 
                                                     y_test_cat, verbose=0)

"""## Plots"""

#Preparing the data for the plots - Translation
translation_magnitude = [2, 4, 8, 16, 28]

accuracies_tr_mlp = [round(i[1]* 100, 3) for i in (score_first_tr_mlp, score_second_tr_mlp, score_third_tr_mlp, 
                                                   score_fourth_tr_mlp, score_fifth_tr_mlp)]

accuracies_tr_cnn = [round(i[1]* 100, 3) for i in (score_first_tr_cnn, score_second_tr_cnn, score_third_tr_cnn, 
                                                   score_fourth_tr_cnn, score_fifth_tr_cnn)]

accuracies_tr_cnn_do = [round(i[1]* 100, 3) for i in (score_first_tr_do, score_second_tr_do, score_third_tr_do, 
                                                      score_fourth_tr_do, score_fifth_tr_do)]

accuracies_tr_cnn_mp = [round(i[1]* 100, 3) for i in (score_first_tr_cnnmp, score_second_tr_cnnmp, 
                                                      score_third_tr_cnnmp, score_fourth_tr_cnnmp, 
                                                      score_fifth_tr_cnnmp)]

accuracies_tr_cnn_ap = [round(i[1]* 100, 3) for i in (score_first_tr_cnnap, score_second_tr_cnnap, 
                                                      score_third_tr_cnnap, score_fourth_tr_cnnap, 
                                                      score_fifth_tr_cnnap)]

accuracies_tr_cnn_gmp = [round(i[1]* 100, 3) for i in (score_first_tr_global_mp, score_second_tr_global_mp, 
                                                       score_third_tr_global_mp, score_fourth_tr_global_mp, 
                                                       score_fifth_tr_global_mp)]  

accuracies_tr_cnn_gap = [round(i[1]* 100, 3) for i in (score_first_tr_global_ap, score_second_tr_global_ap, 
                                                       score_third_tr_global_ap, score_fourth_tr_global_ap, 
                                                       score_fifth_tr_global_ap)]                                                      

accuracies_tr_mlp_aug = [round(i[1]* 100, 3) for i in (score_mlp_first_tr_aug, score_mlp_second_tr_aug, 
                                                       score_mlp_third_tr_aug, score_mlp_fourth_tr_aug, 
                                                       score_mlp_fifth_tr_aug)]

accuracies_tr_cnn_aug = [round(i[1]* 100, 3) for i in (score_cnn_first_tr_aug, score_cnn_second_tr_aug, 
                                                       score_cnn_third_tr_aug, score_cnn_fourth_tr_aug, 
                                                       score_cnn_fifth_tr_aug)]

fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111)
plt.grid(linestyle=':')
plt.plot(translation_magnitude, accuracies_tr_mlp, '-o', color='red', 
         label='Standard MLP')
plt.plot(translation_magnitude, accuracies_tr_cnn, '-o', color= 'blue', 
         label='Standard CNN')
plt.plot(translation_magnitude, accuracies_tr_cnn_do, ':^', color= 'green', 
         label='CNN Dropout')
plt.plot(translation_magnitude, accuracies_tr_cnn_mp, '--s', color='cyan', 
         label='CNN MaxPooling')
plt.plot(translation_magnitude, accuracies_tr_cnn_ap, '-.o', color= 'black', 
         label='CNN AveragePooling')
plt.plot(translation_magnitude, accuracies_tr_cnn_gmp, '--^', color= 'purple', 
         label='CNN Global MaxPooling')
plt.plot(translation_magnitude, accuracies_tr_cnn_gap, '-.s', color= 'magenta', 
         label='CNN Global AveragePooling')
plt.yticks(np.arange(0, 110, step=10))
plt.xticks(np.arange(0, 31, step=2))
ax.set_xlabel('Translation Magnitude')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 Performance of NNs to Translation")
plt.legend(loc='upper right')
plt.savefig('CIFAR-10 Translation.png')
plt.show()

fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111)
#plt.grid(linestyle=':')
plt.plot(translation_magnitude, accuracies_tr_mlp, '-o', color='red', 
         label='Standard MLP')
plt.plot(translation_magnitude, accuracies_tr_mlp_aug, '--^', color='blue', 
         label='MLP Data Augmentation')
plt.plot(translation_magnitude, accuracies_tr_cnn, '-o', color='black', 
         label='Standard CNN')
plt.plot(translation_magnitude, accuracies_tr_cnn_aug, '--^', color='magenta', 
         label='CNN Data Augmentation')
plt.yticks(np.arange(0, 110, step=10))
plt.xticks(np.arange(0, 31, step=2))
ax.set_xlabel('Translation Magnitude')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 NNs Performance Translation")
plt.legend(loc='upper left')
plt.savefig('CIFAR-10 Data Aug Translation.png')
plt.show()
plt.clf()

##Preparing the data for the plots - Rotation
rotation_magnitude = [20, 40, 60, 90]

accuracies_rotation_mlp = [round(i[1] * 100, 3) for i in (score_frotation_mlp, score_srotation_mlp, 
                                                          score_trotation_mlp, score_ftrotation_mlp)]
    
accuracies_rotation_cnn = [round(i[1] * 100, 3) for i in (score_frotation_cnn, score_srotation_cnn,  
                                                          score_trotation_cnn, score_ftrotation_cnn)]

accuracies_rotation_cnn_do = [round(i[1] * 100, 3) for i in (score_frotation_do, score_srotation_do, 
                                                             score_trotation_do, score_ftrotation_do)]

accuracies_rotation_cnn_mp = [round(i[1] * 100, 3) for i in (score_frotation_cnnmp, score_srotation_cnnmp,  
                                                             score_trotation_cnnmp, score_ftrotation_cnnmp)]

accuracies_rotation_cnn_ap = [round(i[1] * 100, 3) for i in (score_frotation_cnnap, score_srotation_cnnap,  
                                                             score_trotation_cnnap, score_ftrotation_cnnap)]
    
accuracies_rotation_cnn_gmp = [round(i[1] * 100, 3) for i in (score_frotation_global_mp, score_srotation_global_mp, 
                                                              score_trotation_global_mp, score_ftrotation_global_mp)]
        
accuracies_rotation_cnn_gap = [round(i[1] * 100, 3) for i in (score_frotation_global_ap, score_srotation_global_ap, 
                                                              score_trotation_global_ap, score_ftrotation_global_ap)]

accuracies_rotation_mlp_aug = [round(i[1] * 100, 3) for i in (score_mlp_frotation_aug, score_mlp_srotation_aug, 
                                                              score_mlp_trotation_aug, score_mlp_ftrotation_aug)]

accuracies_rotation_cnn_aug = [round(i[1] * 100, 3) for i in (score_cnn_frotation_aug, score_cnn_srotation_aug, 
                                                              score_cnn_trotation_aug, score_cnn_ftrotation_aug)]

#Rotation Plots
fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111)
plt.grid(linestyle=':')
plt.plot(rotation_magnitude, accuracies_rotation_mlp, '-o', color='red', 
         label='Standard MLP')
plt.plot(rotation_magnitude, accuracies_rotation_cnn, '-o', color= 'blue', 
         label='Standard CNN')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_do, ':^', color= 'green', 
         label='CNN Dropout')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_mp, '--s', color='cyan', 
         label='CNN MaxPooling')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_ap, '-.o', color= 'black', 
         label='CNN AveragePooling')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_gap, '--^', color= 'purple', 
         label='CNN Global MaxPooling')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_gmp, '-.s', color= 'magenta', 
         label='CNN Global AveragePooling')
plt.yticks(np.arange(0, 110, step=10))
ax.set_xlabel('Rotation Magnitude')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 Performance of NNs to Rotation")
plt.legend(loc='upper right')
plt.savefig('CIFAR-10 NNs Rotation.png')
plt.show()
plt.clf()

fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111)
#plt.grid(linestyle=':')
plt.plot(rotation_magnitude, accuracies_rotation_mlp, '-o', color='red', 
         label='Standard MLP')
plt.plot(rotation_magnitude, accuracies_rotation_mlp_aug, '--^', color= 'blue', 
         label='MLP Data Augmentation')
plt.plot(rotation_magnitude, accuracies_rotation_cnn, '-o', color= 'black', 
         label='Standard CNN Data Augmentation')
plt.plot(rotation_magnitude, accuracies_rotation_cnn_aug, '--^', color= 'magenta', 
         label='CNN Data Augmentation')
plt.yticks(np.arange(0, 110, step=10))
ax.set_xlabel('Rotation Magnitude')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 NNs Performance Rotation")
plt.legend(loc='upper left')

plt.savefig('CIFAR-10 MLPs Rotation.png')
plt.show()
plt.clf()

"""--------------------------------------------------"""

##Preparing the data for the plots - Shearing
shear_magnitude = [20, 35, 50]

accuracies_shear_mlp = [round(i[1] * 100, 3) for i in (score_fshear_mlp, score_sshear_mlp,
                                                       score_tshear_mlp)]                                                   
    
accuracies_shear_cnn = [round(i[1] * 100, 3) for i in (score_fshear_cnn, score_sshear_cnn,  
                                                       score_tshear_cnn)]

accuracies_shear_cnn_do = [round(i[1] * 100, 3) for i in (score_fshear_do, score_sshear_do, 
                                                          score_tshear_do)]

accuracies_shear_cnn_mp = [round(i[1] * 100, 3) for i in (score_fshear_mp, score_sshear_mp,  
                                                          score_tshear_mp)]

accuracies_shear_cnn_ap = [round(i[1] * 100, 3) for i in (score_fshear_ap, score_sshear_ap,  
                                                          score_tshear_ap)]
    
accuracies_shear_cnn_gmp = [round(i[1] * 100, 3) for i in (score_fshear_globalmp, score_sshear_globalmp, 
                                                           score_tshear_globalmp)]
        
accuracies_shear_cnn_gap = [round(i[1] * 100, 3) for i in (score_fshear_globalap, score_sshear_globalap,
                                                           score_tshear_globalap)]

#Shearing Plots
fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
plt.grid(linestyle=':')
plt.plot(shear_magnitude, accuracies_shear_mlp, '-o', color='red', 
         label='Standard MLP')
plt.plot(shear_magnitude, accuracies_shear_cnn, '-o', color= 'blue', 
         label='Standard CNN')
plt.plot(shear_magnitude, accuracies_shear_cnn_do, ':^', color= 'green', 
         label='CNN Dropout')
plt.plot(shear_magnitude, accuracies_shear_cnn_mp, '--s', color='cyan', 
         label='CNN MaxPooling')
plt.plot(shear_magnitude, accuracies_shear_cnn_ap, '-.o', color= 'black', 
         label='CNN AveragePooling')
plt.plot(shear_magnitude, accuracies_shear_cnn_gap, '--^', color= 'purple', 
         label='CNN Global MaxPooling')
plt.plot(shear_magnitude, accuracies_shear_cnn_gmp, '-.s', color= 'magenta', 
         label='CNN Global AveragePooling')
plt.yticks(np.arange(0, 110, step=10))
ax.set_xlabel('Shearing Magnitude')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 Performance of NNs to Shearing")
plt.legend(loc='upper right')
plt.savefig('CIFAR-10 NNs Shearing.png')
plt.show()
plt.clf()

##Preparing the data for the plots - Flipping
flip = ["Horizontal", "Vertical"]

accuracies_flip_mlp = [round(i[1] * 100, 3) for i in (score_hflp_mlp, score_vflp_mlp)]

accuracies_flip_cnn = [round(i[1] * 100, 3) for i in (score_hflp_cnn, score_vflp_cnn)]  

accuracies_flip_cnn_do = [round(i[1] * 100, 3) for i in (score_hflp_do, score_vflp_do)]                                                            

accuracies_flip_cnn_mp = [round(i[1] * 100, 3) for i in (score_hflp_mp, score_vflp_mp)] 

accuracies_flip_cnn_ap = [round(i[1] * 100, 3) for i in (score_hflp_ap, score_vflp_ap)]                                                              

accuracies_flip_cnn_gmp = [round(i[1] * 100, 3) for i in (score_hflp_global_mp, 
                                                          score_vflp_global_mp)] 

accuracies_flip_cnn_gap = [round(i[1] * 100, 3) for i in (score_hflp_global_ap, 
                                                          score_vflp_global_ap)]

#Flip Plots
fig=plt.figure(figsize=(7,5))
ax=fig.add_subplot(111)
plt.grid(linestyle=':')
plt.plot(flip, accuracies_flip_mlp, ':o', color='red', label='Standard MLP')
plt.plot(flip, accuracies_flip_cnn, ':o', color= 'blue', label='Standard CNN')
plt.plot(flip, accuracies_flip_cnn_do, ':^', color= 'green', label='CNN Dropout')
plt.plot(flip, accuracies_flip_cnn_mp, ':s', color='cyan', label='CNN MaxPooling')
plt.plot(flip, accuracies_flip_cnn_ap, ':o', color= 'black', label='CNN AveragePooling')
plt.plot(flip, accuracies_flip_cnn_gap, ':^', color= 'purple', label='CNN Global MaxPooling')
plt.plot(flip, accuracies_flip_cnn_gmp, ':s', color= 'magenta', label='CNN Global AveragePooling')
plt.yticks(np.arange(0, 110, step=10))
ax.set_xlabel('Flip')
ax.set_ylabel('Test Accuracy (%)')
plt.title("CIFAR-10 Performance of NNs to Flipping")
plt.legend(loc='upper left')
plt.savefig('CIFAR-10 NNs Flipping.png')
plt.show()
plt.clf()
