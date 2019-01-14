"""
    @author: lntk
"""

import numpy as np
from scipy.optimize import minimize
import cv2

from script.util import math_utils, image_utils


def contour_distance(box_1, box_2, verbose=False):
    inside_weight = 100000
    intersect_weight = 100000
    distance_weight = 1

    center_1 = get_center(box_1)
    center_2 = get_center(box_2)

    score = np.linalg.norm(center_1[0] - center_2[0]) * distance_weight
    if verbose:
        print("Distance score:", score)

    for i in range(4):
        point = box_1[i][0]
        box_tl = box_2[0][0]
        box_br = box_2[3][0]

        score += inside_weight * math_utils.check_point_inside_rect(point, box_tl, box_br)

    for i in range(4):
        point = box_2[i][0]
        box_tl = box_1[0][0]
        box_br = box_1[3][0]

        score += inside_weight * math_utils.check_point_inside_rect(point, box_tl, box_br)

    segments_1 = [
        [box_1[0][0], box_1[1][0]],
        [box_1[0][0], box_1[2][0]],
        [box_1[1][0], box_1[3][0]],
        [box_1[2][0], box_1[3][0]]
    ]

    segments_2 = [
        [box_2[0][0], box_2[1][0]],
        [box_2[0][0], box_2[2][0]],
        [box_2[1][0], box_2[3][0]],
        [box_2[2][0], box_2[3][0]]
    ]

    for segment_1 in segments_1:
        for segment_2 in segments_2:
            score += intersect_weight * math_utils.check_2_line_segments_intersect(*segment_1, *segment_2)

    if verbose:
        print("Score:", score)
    return score


def get_center(box):
    center = np.add(box[0], box[3])
    center = np.divide(center, 2)
    center = center.astype('int32')
    return center


def translate_contour(contour, shift):
    new_contour = list()
    for point in contour:
        new_point = np.add(point, np.asarray(shift))
        new_contour.append(new_point)
    return np.asarray(new_contour, dtype='int32')


def get_bounding_box_points(contour):
    points = list()
    x, y, w, h = cv2.boundingRect(contour)
    points.append(np.asarray([[x, y]]))
    points.append(np.asarray([[x + w, y]]))
    points.append(np.asarray([[x, y + h]]))
    points.append(np.asarray([[x + w, y + h]]))

    return np.asarray(points)


def generate_chromosome_cluster(contours, verbose=False):
    processed_contours = list()
    processed_boxes = list()  # contains bounding boxes of processed contours
    initial_boxes = list()  # contains bounding boxes of initial contours, used for redraw

    num_contour = len(contours)
    for idx in range(num_contour):
        contour = contours[idx]
        if len(processed_contours) == 0:
            processed_contours.append(contour)
            box = get_bounding_box_points(contour)
            processed_boxes.append(box)
            initial_boxes.append(box)
            continue

        initial_box = get_bounding_box_points(contour)
        initial_boxes.append(initial_box)

        """ Score function representing the closeness and overlapping of chromosomes """

        def score(x):
            # shifted_contour = translate_contour(contour, x)
            shifted_box = translate_contour(initial_box, x)

            f = 0

            for processed_box in processed_boxes:
                f += contour_distance(shifted_box, processed_box)

            return f

        """ Randomize initial shift to one of four corners """
        ids = [-1, 1]
        initial_shift = np.asarray([[10000 * ids[np.random.randint(2)], 10000 * ids[np.random.randint(2)]]])

        """ Optimize positions of chromosomes so that they are close to each other but not overlap """
        result = minimize(score, initial_shift, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        optimized_shift = result.x
        if verbose:
            print("Optimized shift: ", optimized_shift)
        optimized_contour = translate_contour(contour, optimized_shift)

        processed_contours.append(optimized_contour)
        processed_boxes.append(get_bounding_box_points(optimized_contour))

    return processed_contours, processed_boxes, initial_boxes


def get_size(box):
    point_tl = box[0][0]
    point_br = box[3][0]
    return point_br[0] - point_tl[0], point_br[1] - point_tl[1]


def get_chromosome_silhouette(contours, boxes):
    """ Find the minimum rectangle covering all chromosomes (represented by their boxes) """
    min_x, min_y, max_x, max_y = 10000, 10000, -10000, -10000
    for box in boxes:
        point_tl = box[0][0]
        point_br = box[3][0]

        min_x = min(min_x, point_tl[0])
        min_y = min(min_y, point_tl[1])
        max_x = max(max_x, point_br[0])
        max_y = max(max_y, point_br[1])

    """ Shift contours so that they have positive coordinates"""
    shift_x = max(-min_x, 0)
    shift_y = max(-min_y, 0)

    shifted_contours = list()

    for contour in contours:
        shifted_contour = translate_contour(contour, np.asarray([[shift_x, shift_y]]))
        shifted_contours.append(shifted_contour)

    """ Draw contour on a white plane"""
    white_image = 255 - np.zeros((max_y + 200, max_x + 200, 3))
    image = cv2.drawContours(white_image, shifted_contours, contourIdx=-1, color=(255, 0, 0), thickness=cv2.FILLED)

    return image


def get_chromosome_cluster_image(boxes, initial_boxes, initial_chromosomes):
    """ Find the minimum rectangle covering all chromosomes (represented by their boxes) """
    min_x, min_y, max_x, max_y = 10000, 10000, -10000, -10000
    for box in boxes:
        point_tl = box[0][0]
        point_br = box[3][0]

        min_x = min(min_x, point_tl[0])
        min_y = min(min_y, point_tl[1])
        max_x = max(max_x, point_br[0])
        max_y = max(max_y, point_br[1])

    """ Shift contours so that they have positive coordinates"""
    shift_x = max(-min_x, 0) + 100
    shift_y = max(-min_y, 0) + 100

    shifted_boxes = list()

    for box in boxes:
        shifted_box = translate_contour(box, np.asarray([[shift_x, shift_y]]))
        shifted_boxes.append(shifted_box)

    """ Draw chromosomes on a white plane"""
    image = 255 - np.zeros((max_y + shift_y + 100, max_x + shift_x + 100, 3), dtype='int32')
    # print(shift_y, shift_x, max_x + 200, max_y + 200)
    num_chromosome = len(initial_chromosomes)
    for idx in range(num_chromosome):
        i_x1, i_y1, i_x2, i_y2 = shifted_boxes[idx][0][0][0], \
                                 shifted_boxes[idx][0][0][1], \
                                 shifted_boxes[idx][3][0][0], \
                                 shifted_boxes[idx][3][0][1]

        c_x1, c_y1, c_x2, c_y2 = initial_boxes[idx][0][0][0], \
                                 initial_boxes[idx][0][0][1], \
                                 initial_boxes[idx][3][0][0], \
                                 initial_boxes[idx][3][0][1]

        # TODO: This is to fix error of the rounding of function 'translate_contour' making two boxes incompatible
        # Example of error: ValueError: could not broadcast input array from shape (39,40,3) into shape (39,39,3)
        c_x2 = c_x1 + (i_x2 - i_x1)
        c_y2 = c_y1 + (i_y2 - i_y1)
        # w = max(i_x2 - i_x1, c_x2 - c_x1)
        # h = max(i_y2 - i_y1, c_y2 - c_y1)
        # i_x2 = i_x1 + w
        # c_x2 = c_x1 + w
        # i_y2 = i_y1 + h
        # c_y2 = c_y1 + h

        # image_utils.show_multiple_images([image,
        #                                   initial_chromosomes[idx]])

        image[i_y1: i_y2, i_x1: i_x2] = initial_chromosomes[idx][c_y1: c_y2, c_x1: c_x2]

    return image
