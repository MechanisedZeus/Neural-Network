import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import models
from PIL import Image


model = models.load_model("complete_model2/")                                    # loads nural network


def classify(img):
        image = Image.open(img)                                                 # opens image
        image = image.resize((28, 28))                                          # resizes image to 28x28px
        image = np.invert(image.convert('L')).ravel()                           # makes image grey scale
        image = image.reshape(-1, 784).astype("float32") / 255                  # reconfigure the image
        prediction = model.predict(image)                                       # runs the image through the neural network
        prediction = np.argmax(prediction)
        return prediction                                                       # returns the predidction

