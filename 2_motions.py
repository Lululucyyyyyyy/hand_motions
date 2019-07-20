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

#directions = ['five', 'one']
my_list = []
path = "dataset3/*.JPG"
temp = glob.glob(path)
for image in temp:
    with open(image, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224))
        np_img = np.array(img)
        my_list.append(np_img)
my_list = np.array(my_list)
print('my_list.shape: ', my_list.shape)

# 0 five, 1 one
labels = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
print(len(labels))
labels = np.array(labels)

#c = np.c_[my_list.reshape(len(my_list), -1), labels.reshape(len(labels), -1)]
#X_orig = c[:, :my_list.size//len(my_list)].reshape(my_list.shape)
#Y_orig = c[:, my_list.size//len(my_list):].reshape(labels.shape)
p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels[p]
print('shuffled labels:', Y_orig)

X_test_orig = X_orig[0:5]
Y_test_orig = Y_orig[0:5]
#X_dev_orig = X_orig[12:22]
#Y_dev_orig = Y_orig[12:22]
X_train_orig = X_orig[6:len(labels)]
Y_train_orig = Y_orig[6:len(labels)]
print(len(X_train_orig))
print(len(Y_train_orig))
#print(len(X_dev_orig))
#print(len(Y_dev_orig))
print(len(X_test_orig))
print(len(Y_test_orig))

X_train = X_train_orig/255
X_test = X_test_orig/255
X_dev = X_dev_orig/225
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T
Y_dev= convert_to_one_hot(Y_dev_orig, 10).T

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,"W2": W2}
    return parameters

def forward_prop(X, parameters, keep_prob = 0.5):
    W1 = parameters['W1']
    W2 = parameters['W2']
    rate=1-keep_prob
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1,rate=rate)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    A2 = tf.nn.dropout(A2,rate=rate)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    P2 = tf.layers.Flatten()(P2)
    Z3 = tf.keras.layers.Dense(10, activation=None)(P2)
    return Z3

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

 def model(X_train, Y_train, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):
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

        print("Training")
        
        for i in range(len(labels)):
            cost_ = 0
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train})
            cost_ += temp_cost / len(labels)

            if print_cost == True:
                print ("Cost after iteration %i: %f" % (i, cost_))
            if print_cost == True:
                costs.append(cost_)
  
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        predicted = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predicted, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        #print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        #train_accuracy = accuracy(predict_op.eval(), correct_prediction)
        #return train_accuracy, parameters


        saver = tf.train.Saver()
        saver.save(sess, 'hand_motions.ckpt')
        sess.close()
        
    return train_accuracy, parameters

def test(X_test, Y_test, parameters):                                     
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_test.shape[1]                            
    costs = [] 
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()

    print("Validation or Testing")
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(cost, {X: X_test, Y:Y_test})

        plt.plot(np.squeeze(a))
        #plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        #test_accuracy = accuracy(predict_op, correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print(accuracy)
        test_accuracy = accuracy.eval({X: X_train, Y: Y_train})   
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
print('done')
