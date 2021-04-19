###############################################################################
# Function for the VAE model
###############################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2DTranspose, Lambda
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints

class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def conv_block(inputs, filters, kernel_size, strides, name, bn=True):
    outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="{}_Conv1D".format(name))(inputs)
    if bn:
        outputs = InstanceNormalization(name="{}_BatchNorm".format(name))(outputs)
    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU".format(name))(outputs)
    
    return outputs
    
def conv_block_seq(inputs, filters, kernel_size, strides, name, bn=True):
    outputs = conv_block(inputs, filters, kernel_size, strides, name="{}_cb_1".format(name))
    outputs = conv_block(outputs, filters, kernel_size, strides=1, name="{}_cb_2".format(name))
    
    return outputs

def conv_block_seq_res(inputs, filters, kernel_size, strides, name, bn=True, In=True, ResCon=True):
    outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="{}_Conv1D1".format(name))(inputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm1".format(name))(outputs)
    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU1".format(name))(outputs)
    
    outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="{}_Conv1D2".format(name))(outputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm2".format(name))(outputs)
    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU2".format(name))(outputs)
    
    outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="{}_Conv1D3".format(name))(outputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm3".format(name))(outputs)
    
    # Residual Add
    if ResCon:
        res_outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="{}_Conv1Dr".format(name))(inputs)
        if bn:
            res_outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNormr".format(name))(res_outputs)

        outputs = tf.keras.layers.Add()([outputs, res_outputs])
    
    if In:
        outputs = InstanceNormalization(name="{}_InstNorm2".format(name))(outputs)

    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU3".format(name))(outputs)
    
    return outputs

def conv_block_seq_res_fixe(inputs, filters, kernel_size, strides, name, bn=True, In=True, ResCon=True):
    outputs = tf.keras.layers.Conv1D(64, kernel_size, strides=strides, padding="same", name="{}_Conv1D1".format(name))(inputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm1".format(name))(outputs)
    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU1".format(name))(outputs)
    
    outputs = tf.keras.layers.Conv1D(64, 1, strides=strides, padding="same", name="{}_Conv1D2".format(name))(outputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm2".format(name))(outputs)
    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU2".format(name))(outputs)
    
    outputs = tf.keras.layers.Conv1D(256, kernel_size, strides=strides, padding="same", name="{}_Conv1D3".format(name))(outputs)
    if bn:
        outputs = tf.keras.layers.BatchNormalization(name="{}_BatchNorm3".format(name))(outputs)
    
    # Residual Add
    if ResCon:
        outputs = tf.keras.layers.Add()([outputs, inputs])
    
    if In:
        outputs = InstanceNormalization(name="{}_InstNorm2".format(name))(outputs)

    outputs = tf.keras.layers.Activation("relu", name="{}_ReLU3".format(name))(outputs)
    
    return outputs
    

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation=None):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

###############################################################################
# Loss function
###############################################################################



