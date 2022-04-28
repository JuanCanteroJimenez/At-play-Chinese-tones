import tensorflow as tf
import numpy as np


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.strides = strides
        self.filters = filters
        
    def build(self, input_shape):
        self.main_layers = [
            tf.keras.layers.Conv2D(self.filters, 2, strides = self.strides,
            padding="same", use_bias=False, input_shape=input_shape),
            self.activation,
            tf.keras.layers.Conv2D(self.filters, 2, strides = 1,
            padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(self.filters, 1, strides=self.strides,
                padding="same", use_bias=False,input_shape=input_shape),
                tf.keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return( self.activation(Z + skip_Z))

class ResidualUnitTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.strides = strides
        self.filters = filters
        
    def build(self, input_shape):
        self.main_layers = [
            tf.keras.layers.Conv2DTranspose(self.filters, 2, strides = self.strides,
            padding="same", use_bias=False, input_shape=input_shape),
            self.activation,
            tf.keras.layers.Conv2DTranspose(self.filters, 2, strides = 1,
            padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2DTranspose(self.filters, 1, strides=self.strides,
                padding="same", use_bias=False,input_shape=input_shape),
                tf.keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return( self.activation(Z + skip_Z))

class WaveToSpectra(tf.keras.layers.Layer):
    def __init__(self, frame_length=256, frame_step=88, **kwargs):
        super().__init__(**kwargs)
        self.frame_length=frame_length
        self.frame_step = frame_step
        
        self.zero_pad_init = round(frame_length - frame_step)
    def call(self, inputs):
        paddings = tf.constant([[0,0],[self.zero_pad_init, 0]])
        
        input_pad = tf.pad(inputs, paddings)
        z = input_pad
        
        
        result = tf.signal.stft(z, frame_length=self.frame_length, frame_step= self.frame_step, pad_end=True)
        magnitude = tf.math.abs(result)
        power = tf.math.log(magnitude)/tf.math.log(tf.math.reduce_max(magnitude))
        angles = tf.math.angle(result)
        result2 = tf.concat((tf.expand_dims(magnitude,-1), tf.expand_dims(tf.math.sin(angles), -1), tf.expand_dims(tf.math.cos(angles), -1)), axis=3)
        
        return(result2[:, :, 0:128, :,])

class SpectraToWave(tf.keras.layers.Layer):
    def __init__(self, frame_length=256, frame_step=88, x_length=11076, **kwargs):
        super().__init__(**kwargs)
        self.frame_length=frame_length
        self.frame_step = frame_step
        
        self.zero_pad_init = round(frame_length - frame_step)
        self.x_length = x_length
        
        #self.k1 = tf.Variable(41.04331,trainable=True)
    def call(self, inputs):
        z = inputs
        z = tf.pad(z, [[0,0], [0,0], [0,1], [0,0]], "SYMMETRIC")
        magnitude =z[:, :, :, 0]
        
        absolute = magnitude
        z_complex = tf.dtypes.complex(real = absolute*tf.cos(tf.math.atan2(z[:, :, :, 1], z[:, :, :, 2])), imag=absolute*tf.sin(tf.math.atan2(z[:, :, :, 1], z[:, :, :, 2])))
        
        
        
        result = tf.signal.inverse_stft(z_complex, frame_length=self.frame_length, frame_step= self.frame_step, window_fn=tf.signal.inverse_stft_window_fn(self.frame_step))
        result = tf.slice(result, begin=(  0,self.zero_pad_init, ), size=(tf.shape(inputs)[0],self.x_length, ))
        return(((result - tf.math.reduce_min(result))*2)/(tf.math.reduce_max(result)-tf.math.reduce_min(result)))


if __name__ == "__main__":
    data=np.load("data_wave_norm.npy")
    import matplotlib.pyplot as plt
    layer=WaveToSpectra()
    layer2=SpectraToWave()
    predicted = layer(data[2:3])
    predicted2 = layer2(predicted)
    print(predicted.shape)
    plt.plot(predicted2[0])
    plt.show()
    plt.imshow(np.array(predicted[0][ :, :, 0]))
    plt.show()
    