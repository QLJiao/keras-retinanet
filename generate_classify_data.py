import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.compute_overlap import compute_overlap

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

from six import raise_from
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result

def __parse_annotation(element):
    """ Parse an annotation given an XML element.
    """
    truncated = _findNode(element, 'truncated', parse=int)
    difficult = _findNode(element, 'difficult', parse=int)

    class_name = _findNode(element, 'name').text

    box = np.zeros((1, 5))
    box[0, 4] = 0

    bndbox = _findNode(element, 'bndbox')
    box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
    box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
    box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
    box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

    return truncated, difficult, box


def __parse_annotations(xml_root):
    """ Parse all annotations under the xml_root.
    """
    boxes = np.zeros((0, 5))
    for i, element in enumerate(xml_root.iter('object')):
        try:
            truncated, difficult, box = __parse_annotation(element)
        except ValueError as e:
            raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)
        boxes = np.append(boxes, box, axis=0)

    return boxes


def load_annotations(image_name):
    """ Load annotations for an image_index.
    """
    filename = image_name.strip('\n') + '.xml'
    try:
        tree = ET.parse(os.path.join('/home/ustc/jql/VOCdevkit2007/VOC2007', 'Annotations', filename))
        return __parse_annotations(tree.getroot())
    except ET.ParseError as e:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
    except ValueError as e:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)


def compute_annotations(
    anchors,
    annotations,
    negative_overlap=0.2,
    positive_overlap=0.3
):
    """ Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    negative_indices = max_overlaps < negative_overlap

    return positive_indices, negative_indices

def detect(model_name, dataset, score_threshold):
    model_path = os.path.join('.', 'snapshots', model_name+'_pascal_01_7903.h5')
    model = models.load_model(model_path, backbone_name=model_name, convert=True)

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'nodule'}

    img_path = '/home/ustc/jql/VOCdevkit2007/VOC2007/JPEGImages/'
    test_list = []
    with open('/home/ustc/jql/VOCdevkit2007/VOC2007/ImageSets/Layout/' + dataset + '.txt', 'r') as test_file:
        for img_name in test_file:
            test_list.append(img_name)

    result_path = '/home/ustc/jql/x-ray/'+dataset
    negative_path = result_path+'/negative'
    positive_path = result_path+'/positive'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(negative_path)
        os.makedirs(positive_path)

    p_count = 0
    n_count = 0
    for img_name in test_list:
        img_file = img_path + img_name.strip('\n') + '.jpg'
        # print(img_file)
        image = read_image_bgr(img_file)
        gt_boxes = load_annotations(img_name)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # print(boxes)
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale
        th_boxes = []
        img_rois = []
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < score_threshold:
                break
            th_boxes.append(box)
            b = box.astype(int)
            roi = draw[b[1]:b[3]+1,b[0]:b[2]+1].copy()
            img_rois.append(roi)
        th_boxes = np.reshape(th_boxes, (-1,4))
        print(th_boxes)
        positive_indices, negative_indices = compute_annotations(th_boxes.astype(np.float64), gt_boxes.astype(np.float64))

        for index, item in enumerate(positive_indices):
            if(item):
                cv2.imwrite(positive_path+'/'+str(p_count)+'.jpg', img_rois[index])
                p_count += 1

        for index, item in enumerate(negative_indices):
            if(item):
                cv2.imwrite(negative_path + '/' + str(p_count) + '.jpg', img_rois[index])
                n_count += 1

    return


if __name__ == '__main__':
    detect('resnet152', 'test', 0.05)
    detect('resnet152', 'train', 0.05)
    detect('resnet152', 'val', 0.05)
