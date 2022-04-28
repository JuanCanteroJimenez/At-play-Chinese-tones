import glob
from pickletools import optimize

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from sklearn.decomposition import PCA
from aux_functions import create_data, train_step, compute_loss, generate_and_save_images, generate_scatter
from convauto import CVAE

train_images, test_images = create_data()
print(train_images.shape)
print(test_images.shape)

train_size = train_images.shape[0]
batch_size = 32
test_size = test_images.shape[0]


test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))


latent_dim = 100
batch_size=32
num_examples_to_generate=16


random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]


model2 = tf.keras.models.load_model("my_model_pruebas100e.H5")
#generate_and_save_images(model2, 1, test_sample, 100)
generate_scatter(model2, test_images[1:100, :, :, :,])