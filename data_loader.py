import cv2
import numpy as np
import pickle
import os
import glob

save_path = 'data/'

def createClassDictionary():
    class_dictn = {}
    for index, class_label in enumerate(os.listdir('data/training/')):
        class_dictn[class_label] = index
    return class_dictn

def make_dataset(dataType='train'):
    X = []
    y = []
    dataFolder = 'data/{}ing/*/*.png'.format(dataType)
    class_dictn = createClassDictionary()

    for each_file in glob.iglob('data/training/*/*.png'):
        X_current, y_current = cv2.imread(each_file), each_file.split('/')[2]
        X.append(X_current)
        label = class_dictn[y_current]
        y.append(label)

    return np.array(X), np.array(y)


def load_data():
    with open(os.path.join(save_path, "train.pickle"), "rb") as f:
        (X_train, y_train) = pickle.load(f)

    with open(os.path.join(save_path, "val.pickle"), "rb") as f:
        (X_test, y_test) = pickle.load(f)
    
    return X_train/255, y_train, X_test/255, y_test

