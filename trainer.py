import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

model = keras.Sequential(
    [
        keras.Input(shape=784),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
print(model.summary())

model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

n_epochs = 1
print(f"approx time: {round((n_epochs*11 )/60)} minuets")
model.load_weights('complete_model/')
model.fit(x_train, y_train, batch_size=32, epochs=n_epochs, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
model.save('complete_model/')
