import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from PIL import Image
import glob
import random
from tensorflow.keras import layers
import os
import keras.backend as K
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

#directions = ['five', 'one']
def load_images(globpath):
    for image in globpath:
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = img.resize((224, 224))
            np_img = np.array(img)
            my_list.append(np_img)

my_list = []
load_images(glob.glob("dataset3/*.JPG"))
load_images(glob.glob("dataset3/*.jpeg"))
my_list = np.array(my_list)
print('my_list.shape: ', my_list.shape)

# 0 five (74), 1 one (74)
labels = [0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0, 
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1]
print("num labels: ",len(labels))
labels = np.array(labels)

p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels[p]

X_test_orig = X_orig[0:15]
Y_test_orig = Y_orig[0:15]
X_dev_orig = X_orig[16:31]
Y_dev_orig = Y_orig[16:31]
X_train_orig = X_orig[32:len(labels)]
Y_train_orig = Y_orig[32:len(labels)]

X_train = X_train_orig/255
X_test = X_test_orig/255
X_dev = X_dev_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 2).T
Y_test = convert_to_one_hot(Y_test_orig, 2).T
Y_dev= convert_to_one_hot(Y_dev_orig, 2).T

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, 2])
    return X, Y

def initialize_parameters():
    tf.set_random_seed(12)
    #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    W1 = tf.get_variable("W1", [11, 11, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, 8, 256], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 256, 384], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, 384, 384], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W5 = tf.get_variable("W5", [3, 3, 384, 256], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,"W2": W2,"W3": W3,"W4":W4, "W5":W5}
    #W is not actually weights, its the filter size -_-
    return parameters

def forward_prop(X, parameters):
    rate = 0.5
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    Z1 = tf.nn.conv2d(X, W1, strides = [1,4,4,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    #A1 = tf.nn.local_response_normalization(A1, 2, 2e-05, 0,75)
    P1 = tf.nn.max_pool(A1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    #A2 = tf.nn.local_response_normalization(A2, 2, 2e-05, 0,75)
    P2 = tf.nn.max_pool(A2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
    Z3 = tf.nn.conv2d(P2, W3, strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = A3
    Z4 = tf.nn.conv2d(P3, W4, strides=[1,1,1,1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    P4 = A4
    Z5 = tf.nn.conv2d(P4, filter = W5, strides=[1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
    flattened = tf.layers.Flatten()(P5)
    FC6 = tf.keras.layers.Dense(4096, activation='relu')(flattened)
    dropout6 = tf.nn.dropout(FC6, rate=rate)
    FC7 = tf.keras.layers.Dense(4096, activation='sigmoid')(dropout6)
    dropout7 = tf.nn.dropout(FC7, rate=rate)
    FC8 = tf.keras.layers.Dense(2, activation=None)(dropout7)
    return FC8

def compute_cost(Z3, Y, beta=0.001):
    #cost = tf.reduce_mean(cross_entropy(tf.nn.softmax(Z3), Y))
    #m = len(X_train)
    #cost = -(1.0/m) * (np.dot(K.log(tf.nn.softmax(Z3)), tf.transpose(Y)) + np.dot(K.log(1-tf.nn.softmax(Z3)), (1-tf.transpose(Y))))
    
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3,labels=Y))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3,labels=tf.argmax(Y,1)))
    #regularizer = tf.nn.l2_loss(Z3)
    #cost = tf.reduce_mean(cost + beta * regularizer)
    return cost

def model(X_, Y_, learning_rate = 0.009, num_epochs = 10, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()  
    tf.set_random_seed(12)
    seed = 3                               
    (m, n_H0, n_W0, n_C0) = X_train.shape        
    n_y = Y_train.shape [0]                           
    costs = [] 
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)

    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        print("=========Training==========")
        print('147')
        tf.print(cost, output_stream=sys.stdout)
        
        for epoch in range(num_epochs):
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y:Y_train})
            print ("Cost after epoch %i: %f" % (epoch, temp_cost))
            costs.append(temp_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predicted = tf.argmax(Z3, 1)
        correct = tf.argmax(Y, 1)
        correct_prediction = tf.equal(predicted, correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_, Y: Y_})

        #saver = tf.train.Saver()
        #saver.save(sess, '2_motions.ckpt')
        sess.close()
        
    return train_accuracy, parameters

def test(X_, Y_, parameters):                                   
    (m, n_H0, n_W0, n_C0) = X_.shape             
    n_y = Y_.shape    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()

    print("=====Validation or Testing=====")
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(cost, feed_dict={X: X_, Y:Y_})

        plt.plot(np.squeeze(a))
        #plt.show()

        predicted = tf.argmax(Z3, 1)
        correct = tf.argmax(Y, 1)
        correct_prediction = tf.equal(predicted, correct)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_accuracy = accuracy.eval({X: X_, Y: Y_})   
        sess.close()

        return test_accuracy

train, parameters = model(X_train, Y_train)
print("training accuracy: ", train)
dev_accuracy = test(X_dev, Y_dev, parameters)
print("validation accuracy: ", dev_accuracy)
test_accuracy = test(X_test, Y_test, parameters)
print("test accuracy: ", str(test_accuracy))
print('done')
