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

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_valid = to_categorical(y_valid)

print(y_train.shape)

input_shape = X_train.shape[1:]

def createModel():
    model = Sequential()

    model.add(Conv2D(64, (5,5), padding='same', activation='elu', input_shape=(input_shape)))
    model.add(Conv2D(64, (5,5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
    model.add(Conv2D(256, (3,3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
    model.add(Conv2D(128, (3,3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(4096, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = createModel()

model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

batch_size = 128
epochs = 100

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch = int(np.ceil(X_train.shape[0]/float(batch_size))),
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    workers=8
)


print(model.evaluate(X_test, y_test))

model.summary()

import matplotlib.pyplot as plt

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()

model.save('Keras_CNN_Smile.h5')