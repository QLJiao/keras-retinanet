import keras
import os
import numpy as np
import random
import keras.applications.nasnet as nasnet

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, LeakyReLU
from keras import backend as K
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input


dataset_path = '/home/ustc/jql/x-ray'


def img_generator(root_path, dataset, batch_size=4):
    p_r, d, positive_names = os.walk(os.path.join(root_path, dataset, 'positive'))
    n_r, d, negative_names = os.walk(os.path.join(root_path, dataset, 'negative'))
    p_count = len(positive_names)
    n_count = len(negative_names)
    while True:
        p_i = random.randint(0, p_count - 1)
        img1 = image.load_img(os.path.join(p_r, positive_names[p_i]),target_size=(331,331))
        img1 = image.img_to_array(img1)
        x = np.expand_dims(img1, axis=0)

        p_i = random.randint(0, p_count - 1)
        img2 = image.load_img(os.path.join(p_r, positive_names[p_i]), target_size=(331, 331))
        img2 = image.img_to_array(img2)
        x = np.append(x, img2, axis=0)

        p_i = random.randint(0, p_count - 1)
        img3 = image.load_img(os.path.join(p_r, positive_names[p_i]), target_size=(331, 331))
        img3 = image.img_to_array(img3)
        x = np.append(x, img3, axis=0)

        n_i = random.randint(0, n_count - 1)
        img4 = image.load_img(os.path.join(p_r, positive_names[p_i]), target_size=(331, 331))
        img4 = image.img_to_array(img4)
        x = np.append(x, img4, axis=0)

        x = preprocess_input(x)
        y = np.array([1,1,1,0])
        y.reshape((4, 1))
        yield x, y

def train(lr = 1e-5, steps = 20000):
    base_model = nasnet.NASNetLarge(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    activate = LeakyReLU(alpha=0.01)
    x = Dense(1000,activation=activate)(x)
    prediction = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    for layer in base_model.layers:
        layer.trainable = True

    adam = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    model.fit_generator(img_generator(dataset_path, 'train'), steps_per_epoch=steps,
                        validation_data=img_generator(dataset_path, 'val'), validation_steps=200)
    model.save('../../../classify.h5')

if __name__ == '__main__':
    train(lr=1e-5, steps=20000)