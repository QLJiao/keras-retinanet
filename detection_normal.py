import keras
# import keras_retinanet
from keras_retinanet import models
from keras.preprocessing import image
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras.applications.inception_resnet_v2 import preprocess_input
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time
import tensorflow as tf
from generate_classify_data import nms

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

dataset_path = '/home/ustc/jql/x-ray/'

def detect(score_threshold):
    # model_path = os.path.join('.', 'snapshots', 'model_save', 'resnet152_' + str(cross_id) + '_pascal'+'.h5')
    model_path = "/home/ustc/jql/x-ray/keras-retinanet/snapshots/model_save/resnet152_5_pascal.h5"
    model = models.load_model(model_path, backbone_name='resnet152', convert=True)

    test_img_file = '/home/ustc/jql/JSRT/normal.txt'
    test_img_path = '/home/ustc/jql/JSRT/JPEGImages/'

    test_list = []
    with open(test_img_file, 'r') as f:
        for line in f:
            name = line.split('.')[0]
            test_list.append(name)

    result_path = '/home/ustc/jql/x-ray/' + 'jsrt_normal'+str(score_threshold)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    count = 0
    for img_name in test_list:
        img_path = test_img_path + img_name+ '.jpg'
        img = read_image_bgr(img_path)
        print(img_name)

        # copy to draw on
        draw = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.time()

        # preprocess image for network
        img = preprocess_image(img)
        img, scale = resize_image(img)

        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))

        # correct for image scale
        boxes /= scale
        nodule_boxes = []
        color = (0, 0, 0)

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < score_threshold:
                break
            nodule_boxes.append(box)

        nodule_boxes = np.asarray(nodule_boxes)

        if nodule_boxes.shape[0] > 0:
            nodule_boxes = nms(nodule_boxes, 0.1)
            count += len(nodule_boxes)


        for box in nodule_boxes:
            b = box.astype(int)
            draw_box(draw, b, color=color, thickness=10)

        print("processing time: ", time.time() - start)
        cv2.imwrite(result_path+'/'+img_name+'.jpg', draw)

    return count

if __name__ == '__main__':
    count = detect(0.2)
    print(count)
