import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_data(np_file="data256_1.npy", test = 0.20):

    dataset=np.load(np_file)
    
    
    
   
    
    result = []
    for data in dataset:
        #retire NAN
        if(np.sum(np.isnan(data)) > 0):
            pass
        else:
            result.append(data)
    result = np.array(result)
    train = result[0:int(result.shape[0]*(1-test))]
    test = result[int(result.shape[0]*(1-test)):(result.shape[0]-1)]
    return(train, test)

def create_data_Wave(np_file="data_wave_norm.npy", test = 0.20):

    dataset=np.load(np_file)
    
    
    
   
    
    result = []
    for data in dataset:
        #retire NAN
        if(np.sum(np.isnan(data)) > 0):
            pass
        else:
            result.append(data)
    result = np.array(result)
    train = result[0:int(result.shape[0]*(1-test))]
    test = result[int(result.shape[0]*(1-test)):(result.shape[0]-1)]
    return(train, test)

def sample( model, latent_dim,  eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, latent_dim))
    return model.decode(eps, apply_sigmoid=True)

def reparameterize( mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_loss_wave(model, x):
  mean, logvar = model.encode(x)
  z = reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def train_step_wave(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss_wave(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch, test_sample, latent_dim):
  mean, logvar = model.encode(test_sample)
  z = reparameterize(mean, logvar)
  
  predictions = sample(model, latent_dim=latent_dim, eps=z)
  fig = plt.figure(figsize=(16, 16))
  #predictions = tf.image.rgb_to_grayscale(predictions)
  for i in range(predictions.shape[0]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(predictions[i], origin="lower")
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('generated.png')
  plt.show(block=False)
  plt.close()
  fig = plt.figure(figsize=(16, 16))
  #test_sample = tf.image.rgb_to_grayscale(test_sample)
  for i in range(test_sample.shape[0]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(test_sample[i], origin="lower")
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('original.png')
  plt.show(block=False)
  plt.close()
  fig = plt.figure(figsize=(4, 4))

def generate_and_save_images_wave(model, epoch, test_sample, latent_dim):
  mean, logvar = model.encode(test_sample)
  z = reparameterize(mean, logvar)
  
  predictions = sample(model, latent_dim=latent_dim, eps=z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.plot(predictions[i])
    

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('generated.png')
  plt.show(block=False)
  plt.close()
  fig = plt.figure(figsize=(4, 4))


  for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.plot(test_sample[i])
    

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('original.png')
  plt.show(block=False)
  plt.close()

def generate_scatter(model, test_sample):
  mean, logvar = model.encode(test_sample)
  z = reparameterize(mean, logvar)
  pca = PCA(n_components=2)
  encoded_2dd = pca.fit_transform(z)
    
  fig = plt.figure(figsize=(4, 4))

    
  
  plt.scatter(encoded_2dd[:, 0], encoded_2dd[:, 1], cmap='gray', s=0.5, alpha=0.5)
  plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('scatter.png')
  plt.show()
  plt.close()




