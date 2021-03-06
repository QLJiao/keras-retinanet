import keras
# import keras_retinanet
from keras_retinanet import models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
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
from keras_retinanet.bin.classify import train

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
keras.backend.tensorflow_backend.set_session(get_session())


def nms(dets, thresh):
    # boxes 位置
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # boxes scores
    order = np.arange(dets.shape[0])
    areas = np.multiply((x2 - x1 + 1), (y2 - y1 + 1)) # 各 box 的面积
    keep = [] # 记录保留下的 boxes
    while order.size > 0:
        keep.append(dets[order[0]])
        # 计算剩余 boxes 与当前 box 的重叠程度 IoU
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = np.multiply(w, h)
        ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        # 保留 IoU 小于设定阈值的 boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


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
    positive_overlap=0.5
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


def create_dataset(model_name, dataset='train', score_threshold=0.1, content_size=1):
    model_path = os.path.join('.', 'snapshots', model_name+'_pascal_01_7903.h5')
    model = models.load_model(model_path, backbone_name=model_name, convert=True)

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'nodule'}

    img_path = '/home/ustc/jql/VOCdevkit2007/VOC2007/JPEGImages/'
    test_list = []
    with open('/home/ustc/jql/VOCdevkit2007/VOC2007/ImageSets/Layout/' + dataset + '.txt', 'r') as test_file:
        for img_name in test_file:
            test_list.append(img_name)

    result_path = '/home/ustc/jql/x-ray/'+'content_size'+str(content_size)+'/'+dataset
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
            w = b[2]-b[0]
            h = b[3]-b[1]
            y_start = max(b[1]-content_size*h, 0)
            y_end = min(b[3]+content_size*h, draw.shape[0])
            x_start = max(b[0]-content_size*w, 0)
            x_end = min(b[2]+content_size*w, draw.shape[1])
            roi = draw[y_start: y_end, x_start: x_end].copy()
            img_rois.append(roi)

        th_boxes = np.reshape(th_boxes, (-1, 4))
        positive_indices, negative_indices = compute_annotations(th_boxes.astype(np.float64), gt_boxes.astype(np.float64))

        for index_n, item in enumerate(negative_indices):
            if(item):
                cv2.imwrite(negative_path + '/' + str(n_count) + '.jpg', img_rois[index_n])
                n_count += 1
        for index_p, item in enumerate(positive_indices):
            if(item):
                cv2.imwrite(positive_path+'/'+str(p_count)+'.jpg', img_rois[index_p])
                p_count += 1
        print(n_count)
        print(p_count)
    return


if __name__ == '__main__':
    data_path = '/home/ustc/jql/x-ray/content_size0/'
    create_dataset('resnet152', 'train', score_threshold=0.1, content_size=0)
    create_dataset('resnet152', 'val', score_threshold=0.1, content_size=0)
    create_dataset('resnet152', 'test', score_threshold=0.1, content_size=0)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    train(dataset=data_path, base_model=base_model, lr=1e-5, batch=8, epoch=5)