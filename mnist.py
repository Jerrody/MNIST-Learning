from keras import models, layers, losses
from keras.datasets import mnist
# import numpy as np
# import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# maximum accuracy: 0.9978
network = models.Sequential()
network.add(layers.Dense(1024, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                # loss=losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

"""Tried to test model"""
# img = test_images[0]
# img = (np.expand_dims(img,0))

# predictions_single = model.predict(img)

# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# np.argmax(predictions_single[0])
