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
from keras.preprocessing.image import ImageDataGenerator

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.5
    pt1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. -pt1, gamma) * K.log(pt1)) - K.sum(
        (1 - alpha) * K.pow(pt0, gamma) * K.log(1. -pt0))


def train(dataset, base_model, lr=1e-5, batch=4, epoch=1):
    train_flow = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                                    height_shift_range=0.1, zoom_range=0.2)
    train_generator = train_flow.flow_from_directory(dataset+'train', target_size=(299, 299), batch_size=batch)
    num_train = train_generator.samples
    print(num_train)
    print(train_generator.class_indices)

    val_flow = ImageDataGenerator()
    val_generator = val_flow.flow_from_directory(dataset+'val', target_size=(299, 299),batch_size=batch)
    num_val = val_generator.samples

    x = base_model.output
    prediction = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    checkpoint = keras.callbacks.ModelCheckpoint(dataset+'classify_new.h5', monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss')
    model.fit_generator(train_generator, steps_per_epoch=num_train, verbose=1, epochs=epoch,
                        validation_data=val_generator, validation_steps=num_val,
                        callbacks=[checkpoint, early_stopping], shuffle=True)

if __name__ == '__main__':
    dataset_path = '/home/ustc/jql/x-ray/'+'content_size2/'
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    train(dataset=dataset_path, base_model=base_model, lr=1e-5, batch=8, epoch=5)
