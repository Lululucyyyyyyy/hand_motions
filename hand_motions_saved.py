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

with tf.Session() as sess:
	saver.restore(sess, "/hand_motions.ckpt")