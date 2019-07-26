import tensorflow as tf 

converter = tf.lite.TFLiteConverter.from_keras_model_file('the_h5_model.h5')
tflite_model = converter.convert()
open('the_tflite.tflite', 'wb').write(tflite_model)
print('complete')