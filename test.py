import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib

# Matplotlib chooses Xwindows backend by default.
matplotlib.use('Agg')

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

##################### Download Model
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'D:\\github\\models\\research\\object_detection\\test_images\\train.pb\\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'D:\\github\\models\\research\\object_detection\\label_map.pbtxt'

NUM_CLASSES = 1

##################### Load a (frozen) Tensorflow model into memory.
print('Loading model...')
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

##################### Loading label map
print('Loading label map...')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##################### Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##################### Detection
# Path to test image
TEST_IMAGE_PATH = 'D:\\github\\raccoon_dataset\\images\\3.JPEG'

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print('Detecting...')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    print(TEST_IMAGE_PATH)
    image = Image.open(TEST_IMAGE_PATH)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Print the results of a detection.
    print(scores)
    print(classes)
    print(category_index)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    print(TEST_IMAGE_PATH.split('.')[0]+'_labeled.jpg')
    plt.figure(figsize=IMAGE_SIZE, dpi=300)
    plt.imshow(image_np)
    plt.savefig(TEST_IMAGE_PATH.split('.')[0] + '_labeled.jpg')
