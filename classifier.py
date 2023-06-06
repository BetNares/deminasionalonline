import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import ImageFile, Image
from numpy import expand_dims
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf

# TO DO: FIX THE MODEL FILE PATH

# opening and store file in a variable

jsonmodel_path = os.path.dirname(os.path.realpath(__file__)) + '/riceleafdisease_model.json'
h5model_path = os.path.dirname(os.path.realpath(__file__)) + '/riceleafdisease-model.h5'

json_file = open(jsonmodel_path,'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

model = model_from_json(loaded_model_json)

# load weights into new model

model.load_weights(h5model_path)
print("Loaded Model from disk")

# compile and evaluate loaded model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



def getPrediction(img_bytes, model):
    # Loads the image and transforms it to (224, 224, 3) shape
    original_image = Image.open(img_bytes)
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((224, 224), Image.NEAREST)
    
    numpy_image = image.img_to_array(original_image)
    image_batch = expand_dims(numpy_image, axis=0)

    processed_image = preprocess_input(image_batch, mode='caffe')
    preds = model.predict(processed_image)
    
    return preds

def classifyImage(file):
    # Returns a probability scores matrix 
    preds = getPrediction(file, model)
    # Decode tha matrix to the following format (class_name, class_description, score) and pick the heighest score
    # We are going to use class_description since that describes what the model sees
    #prediction = decode_predictions(preds, top=1)
    # prediction[0][0][1] is eqaul to the first batch, top prediction and class_description
    maxnum = np.argmax(preds)
    if maxnum == 0:
        prediction = 'Hispa'
    if maxnum == 1:
        prediction = 'Healthy'
    if maxnum == 2:
        prediction = 'Brown Spot'
    if maxnum == 3:
        prediction = 'Leaf Blast'

    result = str(prediction)
    return result