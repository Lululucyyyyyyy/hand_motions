import subprocess
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

interpreter = tf.lite.Interpreter(model_path='the_tflite.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cam = cv2.VideoCapture(0)
word = ''

num = 0
while True:
	#display camera input
	ret_val, image = cam.read()
	image = cv2.flip(image, 1)
	
	#take image and process
	img = subprocess.call("imagesnap -q -w 0.01 img.jpeg", shell=True)
	img = Image.open("img.jpeg")
	img = img.resize((224, 224))

	input_data = np.array(img)
	input_data = input_data.astype(np.float32)
	input_data = np.expand_dims(input_data, axis=0)

	interpreter.set_tensor(input_details[0]['index'], input_data) 

	toc = time.time()
	interpreter.invoke()
	tic = time.time()
	the_time = tic - toc

	output_data = interpreter.get_tensor(output_details[0]['index'])
	output = np.argmax(output_data)
	if output == 1:
		print('one', the_time)
		word = 'one'
	else:
		print('five', the_time)
		word = 'five'

	#show camera input
	cv2.imshow('my webcam', image)
	if cv2.waitKey(1) == 27: 
		break  # esc to quit

cv2.destroyAllWindows()