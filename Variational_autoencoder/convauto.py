
from resnet import ResidualUnit, ResidualUnitTranspose, SpectraToWave, WaveToSpectra
import tensorflow as tf
import numpy as np
#import tensorflow_probability as tfp

class CVAE(tf.keras.Model):

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Input([128, 128, 3]))
        self.encoder.add(tf.keras.layers.Conv2D(64, 2, strides=1,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        prev_filters=64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.encoder.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self.encoder.add(tf.keras.layers.Conv2D(64, 2, strides=1,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Conv2D(64, 2, strides=1,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))

        



        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Dense(16*16*64, input_shape=[latent_dim, ]))
        self.decoder.add(tf.keras.layers.Reshape([16, 16, 64]))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(512, 2, strides=1, padding="same", use_bias=False))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation("relu"))
        prev_filters= 512
        for filters in [512] * 3 + [256] * 6 + [128] * 4 + [64] * 3:
            strides = 1 if filters == prev_filters else 2
            self.decoder.add(ResidualUnitTranspose(filters, strides=strides))
            prev_filters = filters
        self.decoder.add(tf.keras.layers.Conv2DTranspose(64, 2, strides=1, padding="same", use_bias=False))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(3,2,1,padding="same"))
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def call(self, inputs):
        x = inputs
        mean, logvar = self.encode(x)
        
        return(self.decode(mean))
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.bool)]
    )
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
   

class CVAE_pruebas(tf.keras.Model):

    def __init__(self, latent_dim):
        super(CVAE_pruebas, self).__init__()
        self.latent_dim = latent_dim

        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Input([256, 256, 1]))
        self.encoder.add(tf.keras.layers.Conv2D(64, 2, strides=2,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        prev_filters=64
        for filters in [64] * 4 + [128] * 5 + [256] * 7 + [512] * 4:
            strides = 1 if filters == prev_filters else 2
            self.encoder.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self.encoder.add(tf.keras.layers.Conv2D(512, 2, strides=2,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Conv2D(512, 2, strides=2,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Conv2D(50, 2, strides=2,
        padding="same", use_bias=False))
        self.encoder.add(tf.keras.layers.Flatten())
        

        



        self.decoder = tf.keras.Sequential()
        
        self.decoder.add(tf.keras.layers.Reshape([2, 2, 25], input_shape=[latent_dim, ]))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(25, 2, strides=2,
        padding="same", use_bias=False))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(25, 2, strides=2,
        padding="same", use_bias=False))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
        padding="same", use_bias=False))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation("relu"))
        prev_filters= 512
        for filters in [512] * 4 + [256] * 7 + [128] * 5 + [64] * 4:
            strides = 1 if filters == prev_filters else 2
            self.decoder.add(ResidualUnitTranspose(filters, strides=strides))
            prev_filters = filters
        
        self.decoder.add(tf.keras.layers.Conv2DTranspose(1,2,2,padding="same"))
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def call(self, inputs):
        x = inputs
        mean, logvar = self.encode(x)
        
        return(self.decode(mean))
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.bool)]
    )
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

class CVAE_pruebas_WAVE(tf.keras.Model):

    def __init__(self, latent_dim):
        super(CVAE_pruebas_WAVE, self).__init__()
        self.latent_dim = latent_dim

        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Input([11076,]))
        self.encoder.add(WaveToSpectra())
        self.encoder.add(tf.keras.layers.Conv2D(64, 2, strides=2,
        padding="same", use_bias=False, input_shape=[128, 128, 3]))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(tf.keras.layers.Activation("relu"))
        prev_filters=64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.encoder.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        
        
        
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dim+latent_dim))
        

        



        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Dense(8*8*512, input_shape=[latent_dim, ], activation="relu"))
        self.decoder.add(tf.keras.layers.Reshape([8, 8, 512]))
        
        prev_filters= 512
        for filters in [512] * 3 + [256] * 6 + [128] * 4 + [64] * 3:
            strides = 1 if filters == prev_filters else 2
            self.decoder.add(ResidualUnitTranspose(filters, strides=strides))
            prev_filters = filters
        
        self.decoder.add(tf.keras.layers.Conv2DTranspose(3,2,2,padding="same"))
        self.decoder.add(SpectraToWave())
       
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def call(self, inputs):
        x = inputs
        mean, logvar = self.encode(x)
        
        return(self.decode(mean))
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
    )
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.bool)]
    )
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
   

if __name__ == "__main__":
    from aux_functions import compute_loss_wave
    model = CVAE_pruebas(latent_dim=100)
    
    
    model.encode(np.zeros([1,256, 256,1]))
    model.encoder.summary()
    model.decoder.summary()
    
    


