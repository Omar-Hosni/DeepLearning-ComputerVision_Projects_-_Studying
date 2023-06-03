import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, concatenate, Conv3DTranspose
from tensorflow.keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam

K.set_image_data_format('channels_first')

'''
input layers and its depth
input layer is (num_channels, height, width, length), 
where num_channels you can think of like color channels in an image, 
height, width and length are just the size of the input.
'''
input_layer = Input(shape=(4,160,160,16))

'''
contracting (downward) path
depth0: depth of the first down concolution in the U-net
formula: filters(i) = 32*(2^i) where i is the current depth
layer0:there are two ocnvolution layers for each depth
'''

# Define a Conv3D tensor with 32 filters
down_depth_0_layer_0 = Conv3D(filters=32,
                              kernel_size=(3,3,3),
                              padding='same',
                              strides=(1,1,1))(input_layer)


# Add a relu activation to layer 0 of depth 0
down_depth_0_layer_0 = Activation('relu')(down_depth_0_layer_0)

down_depth_0_layer_1 = Conv3D(filters=32,
                              kernel_size=(3,3,3),
                              padding='same',
                              strides=(1,1,1))(down_depth_0_layer_0)

down_depth_0_layer_1 = Activation('relu')(down_depth_0_layer_1)

# Max pooling
down_depth_0_layer_pool = MaxPooling3D(pool_size=(2,2,2))(down_depth_0_layer_1)

#now for depth 1 layer 0
down_depth_1_layer_0 = Conv3D(filters=64,
                              kernel_size=(3,3,3),
                              strides=(1,1,1))(down_depth_0_layer_pool)

down_depth_1_layer_0 = Activation('relu')(down_depth_1_layer_0)


#depth 1 layer 1
down_depth_1_layer_1 = Conv3D(filters=128,
                              kernel_size=(3,3,3),
                              padding='same',
                              strides=(1,1,1))(down_depth_0_layer_0)

down_depth_1_layer_1 = Activation('relu')(down_depth_1_layer_1)

'''
Expanding (upward) path
we'll use a pool size of (2,2,2) for upsampling.

This is the default value for tf.keras.layers.UpSampling3D
As input to the upsampling at depth 1, you'll use the last layer of the downsampling. In this case, it's the depth 1 layer 1.
'''

up_depth_0_layer_0 = UpSampling3D(size=(2,2,2))(down_depth_1_layer_1)

up_depth_1_concat = concatenate([up_depth_0_layer_0, down_depth_0_layer_1], axis=1)

#Up convolution layer 1
up_depth_1_layer_1 = Conv3D(filters=64,
                            kernel_size=(3,3,3),
                            padding='same',
                            strides=(1,1,1))(up_depth_1_concat)

up_depth_1_layer_1 = Activation('relu')(up_depth_1_layer_1)

#up convolution layer 2
up_depth_1_layer_2 = Conv3D(filters=64,
                            kernel_size=(3,3,3),
                            padding='same',
                            strides=(1,1,1)
                           )(up_depth_1_layer_1)
up_depth_1_layer_2 = Activation('relu')(up_depth_1_layer_2)

# Add a final Conv3D with 3 filters to your network.
final_conv = Conv3D(filters=3, #3 categories
                    kernel_size=(1,1,1),
                    padding='valid',
                    strides=(1,1,1)
                    )(up_depth_1_layer_2)
final_activation = Activation('sigmoid')(final_conv)


model = Model(inputs=input_layer, outputs=final_activation)

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()




