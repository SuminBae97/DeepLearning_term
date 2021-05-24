import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def conv2d(tensor, filters, kernel, padding='same', activation='elu'):
    x = Conv2D(filters=filters, kernel_size=kernel, padding=padding, kernel_initializer = 'he_normal')(tensor)
    x = BatchNormalization()(x)
    x = Activation(activation='elu')(x)

    return x

def convlstm2d(tensor, filters, kernel, padding='same'):
    x = ConvLSTM2D(filters=filters, kernel_size=kernel, padding='same', return_sequences=False)(tensor)
    x = BatchNormalization()(x)

    return x

def getModel(input_shape=(120, 120, 4)):

    inputs = tf.keras.Input(input_shape)

    ### Unet ###
    conv1 = conv2d(inputs, 64, (3,3))           # 120, 120, 64
    conv1 = conv2d(conv1, 64, (3,3))
    conv1 = conv2d(conv1, 64, (3,3))
    conv1_pool = MaxPool2D(pool_size=(2,2))(conv1)  

    conv2 = conv2d(conv1_pool, 128, (3,3))          # 60, 60, 128
    conv2 = conv2d(conv2, 128, (3,3))
    conv2 = conv2d(conv2, 128, (3,3))
    conv1_pool = MaxPool2D(pool_size=(2,2))(conv2)
    
    conv3 = conv2d(conv1_pool, 256, (3,3))          # 30, 30, 256
    conv3 = conv2d(conv3, 256, (3,3))
    conv3 = conv2d(conv3, 256, (3,3))
    conv3_pool = MaxPool2D(pool_size=(2,2))(conv3)
    
    drop1 = SpatialDropout2D(rate=0.25)(conv3_pool) # 15, 15, 256
    
    conv_m = conv2d(drop1, 512, (3,3))              # 15, 15, 512
    conv_m = conv2d(conv_m, 512, (3,3))
    conv_m = conv2d(conv_m, 512, (3,3))
    
    drop2 = SpatialDropout2D(rate=0.25)(conv_m)     # 15, 15, 512

    uconv3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer = 'he_normal')(drop2)
    uconv3 = concatenate([conv3, uconv3], axis=3)
    uconv3 = conv2d(uconv3, 256, (3,3))              # 30, 30, 256
    uconv3 = conv2d(uconv3, 256, (3,3))
    uconv3 = conv2d(uconv3, 256, (3,3))
    
    uconv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer = 'he_normal')(uconv3)
    uconv2 = concatenate([conv2, uconv2], axis=3)
    uconv2 = conv2d(uconv2, 256, (3,3))              # 60, 60, 128
    uconv2 = conv2d(uconv2, 256, (3,3))
    uconv2 = conv2d(uconv2, 256, (3,3))
    
    uconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding="same", kernel_initializer = 'he_normal')(uconv2)
    uconv1 = concatenate([conv1, uconv1], axis=3)
    uconv1 = conv2d(uconv1, 64, (3,3))              # 120, 120, 64
    uconv1 = conv2d(uconv1, 64, (3,3))
    unet_out = conv2d(uconv1, 64, (3,3))
    
    ### LSTM ###
    input_reshape = tf.reshape(tensor=inputs, shape=(-1, inputs.shape[3], inputs.shape[1], inputs.shape[2], 1))
    lstm_out = convlstm2d(input_reshape, filters=16, kernel=(3,3))

    out_concat = concatenate([unet_out, lstm_out], axis=3)  # 120, 120, 80
    out_conv = conv2d(out_concat, 64, (3,3))
    out_conv = conv2d(out_concat, 64, (3,3))
    out_conv = conv2d(out_concat, 64, (3,3))        # 120, 120, 64
    
    outputs = conv2d(out_concat, 1, (3,3))   # 120, 120, 1

    model = Model(inputs=inputs, outputs=outputs)
    # model = Model(inputs=inputs, outputs=unet_out)

    return model

import os
if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = getModel()
    model.summary()