import math
import tensorflow as tf
import os
from PIL import Image
from scipy import ndimage
import glob
import numpy as np

#path for exporting
path = '/dataset3'

#function for rotation
def rotate_images(X_imgs):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (224, 224, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                np_img = np.array(rotated_img)
                my_list.append(np_img)
        
    #X_rotate = np.array(X_rotate, dtype = np.float32)
    #print(X_rotate)
    return my_list

'''
#applying function
my_list = []
temp_list = []
path = "dataset2/*.JPG"
temp = glob.glob(path)
for image in temp:
    with open(image, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224))
        my_list.append(img)
        temp_list.append(img)
        print(len(temp_list))
my_list = rotate_images(temp_list)
my_list = np.array(my_list)
print(len(my_list))
'''
def get_my_list():
    return my_list
