import subprocess
from PIL import Image
import numpy as np
import tensorflow as tf

interpreter = tf.contrib.lite.Interpreter(model_path='the_tflite_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
	img = subprocess.call("imagesnap img.jpeg", shell=True)
	img = img.resize((224, 224))
	img = np.array(image)
	input_data = img
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	prin(output_data)