import os
import random
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

loaded_model = keras.models.load_model("parameters.keras")
img_size = 100


def get_pixels(image_path):
    if image_path[-3:] == "png":
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)

        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        max_dim = tf.reduce_max([height, width])
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2

        img = tf.image.pad_to_bounding_box(img, pad_height, pad_width, max_dim, max_dim)

        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0

        if None not in img:
            return img


uninf_filenames = os.listdir("cell_images/Uninfected")
uninf_data = []
for fn in uninf_filenames[:10]:
    uninf_data.append([get_pixels(f"cell_images/Uninfected/{fn}"), 0])

inf_filenames = os.listdir("cell_images/Parasitized")
inf_data = []
for fn in inf_filenames[:10]:
    inf_data.append([get_pixels(f"cell_images/Parasitized/{fn}"), 1])

data = uninf_data + inf_data
for i in data:
    if None in i:
        data.remove(i)

rndm = random.randrange(len(data))
label = data[rndm][1]
img = data[rndm][0]

prediction = loaded_model(np.expand_dims(img, axis=0))
predicted_label = prediction > 0.5

# plt.imshow(img[0])
# plt.show()
print("Actual:", label)
print("Predicted:", predicted_label.numpy()[0][0].astype(int))
