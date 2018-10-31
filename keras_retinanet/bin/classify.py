import keras
import os
import numpy as np
import random
import glob
import tensorflow as tf

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, LeakyReLU
from keras import backend as K
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

dataset_path = '/home/ustc/jql/x-ray/'


def img_generator(root_path, dataset, batch_size=4):
    positive_names = glob.glob(dataset_path + dataset + '/positive/' + '*.jpg')
    negative_names = glob.glob(dataset_path + dataset + '/negative/' + '*.jpg')
    p_count = len(positive_names)
    n_count = len(negative_names)
    while True:
        p_i = random.randint(0, p_count - 1)
        img1 = image.load_img(positive_names[p_i],target_size=(331,331))
        img1 = image.img_to_array(img1)
        img1 = np.expand_dims(img1, axis=0)
        x = img1

        p_i = random.randint(0, p_count - 1)
        img2 = image.load_img(positive_names[p_i], target_size=(331, 331))
        img2 = image.img_to_array(img2)
        img2 = np.expand_dims(img2, axis=0)
        x = np.append(x, img2, axis=0)

        p_i = random.randint(0, p_count - 1)
        img3 = image.load_img(positive_names[p_i], target_size=(331, 331))
        img3 = image.img_to_array(img3)
        img3 = np.expand_dims(img3, axis=0)
        x = np.append(x, img3, axis=0)

        n_i = random.randint(0, n_count - 1)
        img4 = image.load_img(positive_names[p_i], target_size=(331, 331))
        img4 = image.img_to_array(img4)
        img4 = np.expand_dims(img4, axis=0)
        x = np.append(x, img4, axis=0)

        x = preprocess_input(x)
        y = np.array([1,1,1,0])
        y.reshape((4, 1))

        yield x, y

def train(lr = 1e-5, steps = 20000):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    x = Dense(1000, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    for layer in base_model.layers:
        layer.trainable = True

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    model.fit_generator(img_generator(dataset_path, 'train'), steps_per_epoch=steps, verbose=1,
                        validation_data=img_generator(dataset_path, 'val'), validation_steps=200)
    model.save('../../../classify.h5')

if __name__ == '__main__':
    train(lr=1e-4, steps=20000)
