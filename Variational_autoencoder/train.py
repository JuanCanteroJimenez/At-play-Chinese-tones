import glob
from pickletools import optimize

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from sklearn.decomposition import PCA
from aux_functions import compute_loss_wave, create_data, train_step, compute_loss, generate_and_save_images, create_data_Wave, train_step_wave, generate_and_save_images_wave
from convauto import CVAE, CVAE_pruebas, CVAE_pruebas_WAVE

train_images, test_images = create_data()
print(train_images.shape)
print(test_images.shape)

train_size = train_images.shape[0]
batch_size = 16
test_size = test_images.shape[0]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

optimizer = tf.keras.optimizers.Adam(1e-5)

epochs = 100
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 100
num_examples_to_generate = 4

random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model  = CVAE_pruebas(latent_dim)

history = []
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate]

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  history.append([epoch, elbo])
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  #generate_scatter(model, epoch, test_images)
  generate_and_save_images(model, epoch, test_sample, latent_dim)
  history2 = np.array(history)
  plt.plot(history2[:, 0], history2[:, 1]*-1)
  plt.savefig("history.png")
  plt.show()
  plt.close( )
model(np.zeros([1,256,256,1]))
model.save("my_model_pruebascolors256_100.H5")
  