import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 909
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 80
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

data_dir = 'data/slices/'

data_dir_train = os.path.join(data_dir, 'training')
data_dir_train_image = os.path.join(data_dir_train, 'img')
data_dir_train_mask = os.path.join(data_dir_train, 'mask')

data_dir_test = os.path.join(data_dir, 'test')
data_dir_test_image = os.path.join(data_dir_train, 'img')
data_dir_test_mask = os.path.join(data_dir_train, 'mask')

NUM_TRAIN = 360
NUM_TEST = 100

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255, rotation_range=90)
    img_datagen = ImageDataGenerator(**data_gen_args)
    msk_datagen = ImageDataGenerator(**data_gen_args)

    img_generator = img_datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = img_datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)

    return zip(img_generator, msk_generator)

train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_train(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)


def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['input image', 'true mask', 'predicted mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(tf.keras.preprocessing.array_to_img(display_list[i]), cmap='gray')

    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0, num):
        image, mask=  next(datagen)
        display({image[0], mask[0]})

show_dataset(train_generator, 2)

def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='name')

    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = tf.keras.layers.Conv2D(initial_features*2**level, **convpars)(x)

        if level < n_levels - 1:
            skips[level] = x
            x = tf.keras.layers.MaxPool2D(pooling_size)(x)

    #upstream
    for level in reversed(range(n_levels-1)):
        x = tf.keras.layers.Conv2DTranspose(initial_features*2**level, strides=pooling_size, **convpars)(x)
        x = tf.keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = tf.kersa.layers.Conv2D(initial_features * 2** level, **convpars)(x)

    x = tf.keras.layers.Conv2D(out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')



EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

model = unet()
model.compile(optimize='adam', loss='binary_crossentropy', matric=['accuracy'])
model.fit_generator(generator=train_generator,
                    steps_per_epoch=EPOCH_STEP_TRAIN,
                    validation_data=test_generator,
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=100)

model.save(f'UNET-Tooth segmentation_{IMAGE_HEIGHT}_{IMAGE_WIDTH}.h5')

test_generator = create_segmentation_generator_train(data_dir_test_image, data_dir_test_mask,1)
def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = model.predict(image)[0]
        display({image[0], mask[0], pred_mask})


show_prediction(test_generator, 3)
