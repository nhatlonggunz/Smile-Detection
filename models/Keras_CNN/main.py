import numpy as np

import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from sklearn.model_selection import train_test_split

import preprocess

X, y = preprocess.read_data('../../smiles')
X = preprocess.normalize(X)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess.split_data(X,y)

input_shape = X_train.shape[1:]
print(input_shape)
print(X_train.shape)

def createModel():
    model = Sequential()

    model.add(Conv2D(64, (11,11), padding='same', activation='relu', input_shape=(input_shape)))
    model.add(Conv2D(64, (11,11), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (9,9), padding='same', activation='relu'))
    model.add(Conv2D(128, (9,9), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (7,7), padding='same', activation='relu'))
    model.add(Conv2D(64, (7,7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))

    return model

model = createModel()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

batch_size = 128
epochs = 50

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch = int(np.ceil(X_train.shape[0]/float(batch_size
    ))),
    epochs=epochs,
    validation_data=(X_valid, y_valid)
)
model.evaluate(X_test, y_test)

model.save('Keras_CNN_Smile.h5')