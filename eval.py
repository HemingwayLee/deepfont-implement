# from matplotlib.pyplot import imshow
# import matplotlib.cm as cm
# import matplotlib.pylab as plt
import os
import random
import PIL
import cv2
import argparse
import itertools
import numpy as np
from imutils import paths
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks, optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K

def rev_conv_label(label):
    if label == 0 :
        return 'Lato'
    elif label == 1:
        return 'Raleway'
    elif label == 2 :
        return 'Roboto'
    elif label == 3 :
        return 'Sansation'
    elif label == 4:
        return 'Walkway'

if __name__ == '__main__':
    img_path="sample/sample.jpg"
    pil_im =PIL.Image.open(img_path).convert('L')
    org_img = img_to_array(pil_im)
    data=[]
    data.append(org_img)
    data = np.asarray(data, dtype="float") / 255.0

    model = load_model('top_model.h5')
    y = model.predict_classes(data)
    label = rev_conv_label(int(y[0]))

    print(f"{img_path}: {label}")
