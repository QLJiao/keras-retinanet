import keras
import os
import numpy as np
import glob
import shutil
import tensorflow as tf

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, LeakyReLU, concatenate, Input, Dropout
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

dataset_path = '/home/ustc/jql/jsrtdata/'
# model_path = '/home/ustc/jql/all_lung_image/'


# def prepare_train(cls='abnormal'):
#
#     file_names = glob.glob(dataset_path+cls+'/*.png')
#     train_cls_path = dataset_path+'train/'+cls
#     test_cls_path = dataset_path+'test/'+cls
#
#     if not os.path.exists(train_cls_path):
#         os.mkdir(train_cls_path)
#     if not os.path.exists(test_cls_path):
#         os.mkdir(test_cls_path)
#     for index, file in enumerate(file_names):
#         dst_name = file.split('/')[-1]
#         if index % 30 == 0:
#             print(dst_name)
#             shutil.copyfile(file, os.path.join(test_cls_path, dst_name))
#         else:
#             shutil.copyfile(file, os.path.join(train_cls_path, dst_name))


def train(lr=1e-3, batch=16, epoch=1):
    train_flow = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, shear_range=0.2,
                                    height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_flow.flow_from_directory(dataset_path+'train', target_size=(299, 299),
                                                     classes=['normal', 'nodule'],
                                                     interpolation='bilinear',
                                                     batch_size=batch)
    num_train = train_generator.samples
    print(num_train)
    print(train_generator.class_indices)

    val_flow = ImageDataGenerator()
    val_generator = val_flow.flow_from_directory(dataset_path+'test', target_size=(299, 299),
                                                 classes=['normal', 'nodule'],
                                                 interpolation='bilinear',
                                                 batch_size=batch)
    num_val = val_generator.samples
    # base_model1 = NASNetLarge(weights='imagenet', include_top=True, input_tensor=input)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.3)(x)
    prediction = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    # model.load_weights('imagenet', by_name=True, skip_mismatch=True)

    for layer in base_model.layers:
        layer.trainable = False

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    checkpoint = keras.callbacks.ModelCheckpoint(dataset_path+'cxr8_jsrt.h5',
                                                 monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss')

    model.fit_generator(train_generator, steps_per_epoch=num_train, verbose=1, epochs=epoch,
                        validation_data=val_generator, validation_steps=int(num_val/batch),
                        callbacks=[checkpoint, early_stopping], shuffle=True)

    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    model.fit_generator(train_generator, steps_per_epoch=num_train, verbose=1, epochs=epoch,
                        validation_data=val_generator, validation_steps=int(num_val / batch),
                        callbacks=[checkpoint, early_stopping], shuffle=True)


if __name__ == '__main__':
    # prepare_train(cls='normal')
    train(lr=1e-4, batch=1, epoch=5)
