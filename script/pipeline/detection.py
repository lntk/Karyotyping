"""
    @author: lntk
"""

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from util import image_utils

# import miscellaneous modules
import numpy as np
import time
import tensorflow as tf
from os.path import dirname, abspath

model_dir = dirname(dirname(dirname(__file__))) + "/model"


def load_model(model_path='default'):
    # set tf backend to allow memory to grow, instead of claiming everything
    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    if model_path == 'default':
        # working_dir = dirname(dirname(dirname(abspath("x"))))
        # model_dir = working_dir + "/model"
        # model_path = model_dir + "/default_inference.h5"
        model_path = model_dir + "/default_inference.h5"

    model = models.load_model(model_path, backbone_name='resnet50')
    return model


def detect_interesting_points(chromosome, model, verbose=False):
    labels_to_names = {0: 'center', 1: 'p', 2: 'mid_p', 3: 'q', 4: 'mid_q_1', 5: 'mid_q_2'}

    # copy to draw on
    draw = chromosome.copy()

    # preprocess image for network
    image = preprocess_image(chromosome)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    if verbose:
        print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    chosen_boxes = [None, None, None, None, None, None]
    counter = 0

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if label < 0:
            break

        if verbose:
            print(box, score, label)

        #     # scores are sorted so we can break
        #     if score < 0.4:
        #         break

        if chosen_boxes[label] is None:
            chosen_boxes[label] = [box, score, label]
            counter += 1
        else:
            continue

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

        if counter >= 6:
            break

    if verbose:
        image_utils.show_image(draw, cmap=None)

    interesting_points = list()
    for label in range(6):
        if chosen_boxes[label] is None:
            continue
        x1, y1, x2, y2 = chosen_boxes[label][0]
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        interesting_points.append([x, y])

    image_with_points = image_utils.get_image_with_points(chromosome, interesting_points, is_scaled=False)
    if verbose:
        image_utils.show_image(image_with_points)

    return interesting_points, draw, image_with_points
