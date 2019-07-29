import subprocess
from PIL import Image
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='the_tflite.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
	img = subprocess.call("imagesnap -q -w 1.00 img.jpeg", shell=True)
	img = Image.open("img.jpeg")
	img = img.resize((224, 224))

	input_data = np.array(img)
	input_data = input_data.astype(np.float32)
	input_data = np.expand_dims(input_data, axis=0)

	interpreter.set_tensor(input_details[0]['index'], input_data) 

	interpreter.invoke()

	output_data = interpreter.get_tensor(output_details[0]['index'])
	output = np.argmax(output_data)
	if output == 1:
		print('one')
	else:
		print('five')