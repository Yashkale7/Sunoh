import pickle
import tensorflow as tf

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

model.save('my_model.keras')


# filename = 'savedmodel.pkl'
# model = pickle.load(open(filename, 'rb'))
# converter = tf.lite.TFLiteConverter.from_saved_model(model)
# tflite_model = converter.convert()