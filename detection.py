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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

dataset_path = '/home/ustc/jql/x-ray/'

def detect(cross_id, score_threshold):
    model_path = os.path.join('.', 'snapshots', 'resnet152_' + str(cross_id) + '_pascal'+'.h5')
    model = models.load_model(model_path, backbone_name='resnet152', convert=True)

    test_img_path = '/home/ustc/jql/JSRT/JPEGImages/'
    test_list = []
    with open('/home/ustc/jql/JSRT/ImageSets/Main/' + 'test' + str(cross_id) + '.txt', 'r') as test_file:
        for img_name in test_file:
            test_list.append(img_name)

    result_path = '/home/ustc/jql/x-ray/' + 'jsrt_result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for img_name in test_list:
        img_path = test_img_path + img_name.strip('\n') + '.jpg'
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
        color = (0, 255, 0)

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < score_threshold:
                break
            nodule_boxes.append(box)

        nodule_boxes = np.asarray(nodule_boxes)
        nodule_boxes = nms(nodule_boxes, 0.15)

        for box in nodule_boxes:
            b = box.astype(int)
            draw_box(draw, b, color=color, thickness=10)
        # for box in nodule_boxes:
        #     b = box.astype(int)
        #     if classify:
        #         w = b[2] - b[0]
        #         h = b[3] - b[1]
        #         roi = draw[b[1]-h:b[3] + h, b[0]-w:b[2] + w].copy()
        #         size = roi.size
        #         if size > 0:
        #             roi = cv2.resize(roi, (299, 299))
        #             roi = np.expand_dims(roi, axis=0)
        #             pred = model_cls.predict(roi)
        #             if pred[0, 1] > 0.4:
        #                 print(pred)
        #                 draw_box(draw, b, color=color, thickness=10)
        #
        #     else:
        #         draw_box(draw, b, color=color, thickness=10)
        print("processing time: ", time.time() - start)
        cv2.imwrite(result_path+'/'+img_name+'.jpg', draw)
    return

if __name__ == '__main__':
    for cross_id in range(10, 15):
        detect(cross_id, 0.075)
