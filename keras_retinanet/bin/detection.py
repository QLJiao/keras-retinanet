import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def detect(model_name, score_threshold):
    model_path = os.path.join('.', 'snapshots', model_name+'_pascal_01.h5')
    model = models.load_model(model_path, backbone_name=model_name, convert=True)

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'nodule'}

    test_img_path = '/home/ustc/jql/VOCdevkit2007/VOC2007/JPEGImages/'
    test_list = []
    with open('/home/ustc/jql/VOCdevkit2007/VOC2007/ImageSets/Layout/test.txt', 'r') as test_file:
        for img_name in test_file:
            test_list.append(img_name)

    result_path = '/home/ustc/jql/keras-retinanet/'+model_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for img_name in test_list:
        img_path = test_img_path + img_name.strip('\n') + '.jpg'
        image = read_image_bgr(img_path)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < score_threshold:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color, thickness=10)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        cv2.imwrite(result_path+'/'+img_name+'.jpg', draw)
    return

if __name__ == '__main__':
    detect('resnet152', 0.2)
