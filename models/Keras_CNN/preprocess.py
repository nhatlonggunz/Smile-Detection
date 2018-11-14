import cv2
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

face_target_size = (48,48)

from os import listdir

def read_data(path):
    posdir = path + '/pos'
    negdir = path + '/neg'

    pos = listdir(posdir)
    neg = listdir(negdir)

    image = np.zeros([len(pos) + len(neg), face_target_size[0], face_target_size[1]])
    label = np.zeros([len(pos) + len(neg), 1])

    i = 0
    for p in pos:
        pic = cv2.imread(posdir + '/' + p, cv2.IMREAD_GRAYSCALE)
        image[i,:,:] = pic
        label[i,:] = 1
        
        i += 1

    for p in neg:
        pic = cv2.imread(negdir + '/' + p, cv2.IMREAD_GRAYSCALE)
        image[i,:,:] = pic

        i += 1

    permu = np.random.permutation(image.shape[0])

    image = image[permu]
    label = label[permu]

    # image = image/255

    return image, label

def normalize(image):
    image /= 255
    return image

def split_data(X, y):
    from sklearn.model_selection import train_test_split
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5)

    return X_train, X_valid, X_test, y_train, y_valid, y_test