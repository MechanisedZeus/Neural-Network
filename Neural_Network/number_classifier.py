import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import models
from PIL import Image

model = models.load_model("complete_model/")

image_number = 1
while os.path.isfile(f"test_data/digit{image_number}.png"):
    try:
        img = np.invert(Image.open(f"test_data/digit{image_number}.png"f"").convert('L')).ravel()
        img = img.reshape(-1, 784).astype("float32") / 255
        prediction = model.predict(img)
        print(np.argmax(prediction))
        # plt.imshow(img[0], cmap=plt.cm.binary)
        # plt.show()
    finally:
        image_number += 1
