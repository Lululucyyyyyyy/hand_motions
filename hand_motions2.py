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
import glob
import random
from tensorflow.keras import layers

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rotate_and_add_image(img):
    ninety = img.rotate(90)
    oneeighty = img.rotate(180)
    twoseventy = img.rotate(270)
    my_list.append(np.array(img))
    my_list.append(np.array(ninety))
    my_list.append(np.array(oneeighty))
    my_list.append(np.array(twoseventy))

    #X_rotate = np.array(X_rotate, dtype = np.float32)
    #print(X_rotate)

#directions = ['up', 'down', 'left', 'right']
my_list = []
path = "dataset2/*.JPG"
temp = glob.glob(path)
for image in temp:
    with open(image, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224))
        rotate_and_add_image(img)
my_list = np.array(my_list)
print("num images: ", len(my_list))

# 0 up, 1 down, 2 left, 3 right,
labels = [1,2,0,3, 1,2,0,3, 1,2,0,3, 1,2,0,3, 1,2,0,3, 1,2,0,3, 
        1,2,0,3, 1,2,0,3, 1,2,0,3, #9 down first
        2,0,3,1, 2,0,3,1, 2,0,3,1, 2,0,3,1, 2,0,3,1, 2,0,3,1, 
        2,0,3,1, 2,0,3,1, 2,0,3,1, #9 left first
        3,1,2,0, 3,1,2,0, 3,1,2,0, 3,1,2,0, 3,1,2,0, 3,1,2,0, 
        3,1,2,0, 3,1,2,0, 3,1,2,0, #9 right first
        0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2, 
        0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2, #9 up first
         ]
print("num labels: ", len(labels))
labels = np.array(labels)

p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels[p]

X_test_orig = X_orig[0:30]
Y_test_orig = Y_orig[0:30]
X_dev_orig = X_orig[31:61]
Y_dev_orig = Y_orig[31:61]
X_train_orig = X_orig[62:len(labels)]
Y_train_orig = Y_orig[62:len(labels)]

X_train = X_train_orig/255
X_test = X_test_orig/255
X_dev = X_dev_orig/225
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T
Y_dev = convert_to_one_hot(Y_dev_orig, 10).T

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed=0), regularizer="l2")
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed=0), regularizer="l2")
    W3 = tf.get_variable("W3", [2, 2, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed=0), regularizer="l2")
    W4 = tf.get_variable("W4", [2, 2, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed=0), regularizer="l2")
    parameters = {"W1": W1,"W2": W2,"W3": W3,"W4":W4}
    return parameters

def forward_prop(X, parameters, keep_prob = 0.5):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    rate=1-keep_prob
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1,rate=rate)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    A2 = tf.nn.dropout(A2,rate=rate)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    #Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
    #A3 = tf.nn.relu(Z3)
    #A3 = tf.nn.dropout(A3,rate=rate)
    #P3 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    #Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = 'SAME')
    #A4 = tf.nn.relu(Z4)
    #A4 = tf.nn.dropout(A4,rate=rate)
    #P4 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,2,2,1], padding = 'SAME')
    P4 = tf.layers.Flatten()(P2)
    Z5 = tf.keras.layers.Dense(10, activation=None)(P4)
    return Z5

def compute_cost(Z3, Y, beta=0.01):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3,labels=Y))
    regularizer = tf.nn.l2_loss(Z3)
    cost = tf.reduce_mean(cost + beta * regularizer)
    #cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(Z3,1e-10,1.0)))
    #cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(Z3,1e-10,1.0)))
    #labels_sum = tf.reduce_sum(Y, axis=-1)
    #print(labels)
    #softmax = tf.nn.softmax(Z3)
    #cost = tf.reduce_mean(-tf.reduce_sum(softmax * tf.log(Y), axis=-1))
    return cost

def model(X_train, Y_train, learning_rate = 0.009, print_cost = True):
    ops.reset_default_graph()                         
    tf.set_random_seed(1)
    seed = 3                                          
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = [] 
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print("==========Training==========")
        
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
  
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        #train_accuracy = tf.metrics.accuracy(labels=Y_train, predictions=X_train)
        #train_accuracy = accuracy(predict_op.eval(), correct_prediction)
        #return train_accuracy, parameters


        saver = tf.train.Saver()
        saver.save(sess, 'hand_motions.ckpt')
        sess.close()
        
    return train_accuracy, parameters

def test(X_, Y_, parameters):                                     
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_.shape[1]                            
    costs = [] 
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()


    print("========Validating or Testing========")
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(cost, {X: X_, Y:Y_})

        #plt.plot(np.squeeze(a))
       # plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        #test_accuracy = accuracy(predict_op, correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_accuracy = accuracy.eval({X: X_, Y: Y_})   
        #test_accuracy = tf.metric.accuracy(labels=Y_test, predictions=X_test)
        saver = tf.train.Saver()
        saver.save(sess, 'hand_motions.ckpt')
        sess.close()
    return test_accuracy

train, parameters = model(X_train, Y_train)
print("training accuracy: ", train)
dev = test(X_dev, Y_dev, parameters)
print("validation accuracy: ", str(dev))
test = test(X_test, Y_test, parameters)
print("test accuracy: ", str(test))
print('==========done==========')

