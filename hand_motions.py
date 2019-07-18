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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#directions = ['up', 'down', 'left', 'right', 'zoom_in', 'zoom_out', 'two', 'three', 'four', 'five']
my_list = []
path = "dataset/*.JPG"
temp = glob.glob(path)
for image in temp:
    with open(image, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224))
        np_img = np.array(img)
        my_list.append(np_img)
my_list = np.array(my_list)
print(my_list.shape)
# 0 up, 1 down, 2 left, 3 right, 4 zoom in, 5 zoom out, 6 two, 7 three, 8 four, 9 five
labels = [0,0,0,0,0,0,0,0,0,0,0,0,
         1,1,1,1,1,1,1,1,1,1,1,
         2,2,2,2,2,2,2,2,2,2,2,2,
         3,3,3,3,3,3,3,3,3,3,3,3,
         4,4,4,4,4,4,4,4,4,4,4,4,
         5,5,5,5,5,5,5,5,5,5,5,5,
         6,6,6,6,6,6,6,6,6,6,6,
         7,7,7,7,7,7,7,7,7,7,7,
         8,8,8,8,8,8,8,8,8,8,8,8,
         9,9,9,9,9,9,9,9,9,9,9,9]
print(len(labels))
labels = np.array(labels)
c = np.c_[my_list.reshape(len(my_list), -1), labels.reshape(len(labels), -1)]
X_orig = c[:, :my_list.size//len(my_list)].reshape(my_list.shape)
Y_orig = c[:, my_list.size//len(my_list):].reshape(labels.shape)
X_test_orig = X_orig[0:10]
Y_test_orig = Y_orig[0:10]
X_train_orig = X_orig[11:117]
Y_train_orig = Y_orig[11:117]
print(len(X_train_orig))
print(len(Y_train_orig))
print(len(X_test_orig))
print(len(Y_test_orig))
X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
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

def forward_prop(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    P2 = tf.layers.Flatten()(P2)
    Z3 = tf.keras.layers.Dense(10, activation=None)(P2)
    return Z3

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits_v2(logits=Z3,labels=Y))
    #cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(Z3,1e-10,1.0)))
    #cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(Z3,1e-10,1.0)))
    #labels_sum = tf.reduce_sum(Y, axis=-1)
    #print(labels)
    #softmax = tf.nn.softmax(Z3)
    #cost = tf.reduce_mean(-tf.reduce_sum(softmax * tf.log(Y), axis=-1))
    return cost

'''
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")'''
#saver = tf.train.Saver()


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):
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
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
                

            ckpt.step.assign_add(1)

            if print_cost == True and epoch % 5 == 0:
                #save_path = manager.save()
                #print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
  
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        #saver.save(sess, 'my_test_model',global_step = 50)

        return train_accuracy, test_accuracy, parameters
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
print(done)




