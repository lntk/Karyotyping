import numpy as np
import cv2
from matplotlib import pyplot as plt
from script.util import math_utils
from skimage.morphology import reconstruction
from os.path import dirname

data_dir = dirname(dirname(dirname(__file__))) + "/data"


def extract_object_by_contour(gray, contours, component_size=256):
    """
    :param gray: an 2d numpy array of an image
    :param contours: a list of contour
    :param component_size: width/height of object images
    :return: a list of 2d numpy array of object images
    """
    ret, labels = contours_to_connected_components(gray, contours)
    height, width = gray.shape
    ret, labels, points, area = connected_component_with_stats(ret, labels)

    object_images = list()

    for label in range(1, ret):
        # can replace by bounding rectangle
        x_points = [x for (x, y) in points[label]]
        y_points = [y for (x, y) in points[label]]
        x_min, x_max, y_min, y_max = min(x_points), max(x_points), min(y_points), max(y_points)

        white_image = 255 - np.zeros_like(gray)
        white_image[labels == label] = gray[labels == label]
        left_x = int((x_max + x_min - component_size) / 2)
        right_x = int((x_max + x_min + component_size) / 2)
        left_y = int((y_max + y_min - component_size) / 2)
        right_y = int((y_max + y_min + component_size) / 2)
        if left_x >= 0 and right_x < height and left_y >= 0 and right_y < width:
            object_image = math_utils.sub_matrix(white_image, left_x, right_x, left_y, right_y)
            object_images.append(object_image)
        else:
            raise Exception("Image is not fitted.")

    return object_images


def extract_object_by_contour_box(gray, contours):
    """
    This function uses the bounding box of the contour of an object to extract the image of that object.
    It processes on an image containing many objects and those corresponding contours

    :param gray: an 2d numpy array of an image
    :param contours: a list of contour
    :return: a list of 2d numpy array of object images
    """
    object_images = list()
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        object_image = gray[x: x + width, y: y + height]
        object_images.append(object_image)
    return object_images


def contours_to_connected_components(gray, contours):
    # GET THE IMAGE WITH CONTOURS AS COMPONENTS
    labels = np.zeros_like(gray)  # dark image
    ret = 1
    for contour in contours:
        labels = cv2.drawContours(labels, [contour], contourIdx=-1, color=(ret, ret, ret), thickness=cv2.FILLED)
        ret += 1

    return ret, labels


def connected_component_with_stats(ret, labels):
    # ret, labels = cv2.connectedComponents(binary)
    points = dict()
    area = dict()
    for i in range(0, ret):
        points[i] = list()
        area[i] = 0
    height, width = labels.shape
    for i in range(height):
        for j in range(width):
            points[labels[i][j]].append((i, j))
            area[labels[i][j]] += 1

    return ret, labels, points, area


def match_shape(gray1, gray2):
    hu_moment_1 = cv2.HuMoments(cv2.moments(gray1)).flatten()
    hu_moment_2 = cv2.HuMoments(cv2.moments(gray2)).flatten()
    # print(hu_moment_1)
    # print(hu_moment_2)
    # m_1 = np.sign(hu_moment_1) * np.log(hu_moment_1)
    # m_2 = np.sign(hu_moment_2) * np.log(hu_moment_2)
    matching_value = np.sum(np.abs(hu_moment_1 - hu_moment_2))
    return matching_value


# to use SIFT, using opencv 3.4.2
# pip install opencv-python=3.4.2
# pip install opencv_contrib-python=3.4.2
def sift_matching(gray1, gray2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print(kp1)
    print(kp2)

    good = []
    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good.append([m])

    matching_image = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good, gray2.copy(), flags=2)
    show_image(matching_image)


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def show_image(image, name='image', cmap='gray', shower='pyplot'):
    if shower == "pyplot":
        plt.grid(False)
        # plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image, cmap=cmap), plt.show()
        return

    if shower == "cv":
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


def show_multiple_images(images, name='image', cmap='gray', shower='pyplot', mode='different'):
    if len(images) > 5:
        print("Only show 5 images.")
        images = images[:5]

    if shower == "pyplot":
        if mode == "same":
            fig = plt.figure()
            num_image = len(images)
            for i in range(num_image):
                plt.grid(False)
                fig.add_subplot(i)
                plt.imshow(images[i])
            plt.show()
            return
        if mode == "different":
            idx = 1
            for image in images:
                plt.figure(idx)
                plt.grid(False)
                plt.imshow(image, cmap=cmap)
                idx += 1
            plt.show()
            return


def get_contour_image(contours, shape):
    white_image = 255 - np.zeros(shape)
    contour_image = cv2.drawContours(white_image, contours, contourIdx=-1, color=(0, 0, 0), thickness=1)
    return contour_image


def get_image_with_contours(image, contours, thickness=1):
    contour_image = cv2.drawContours(image, contours, contourIdx=-1, color=(0, 0, 255), thickness=thickness)
    return contour_image


def unsharp_masking(image, radius=5, mask_weight=10):
    gaussian = cv2.GaussianBlur(image, (radius, radius), sigmaX=1)
    unsharp_image = cv2.addWeighted(image, 1 + mask_weight, gaussian, - mask_weight, gamma=0)
    return unsharp_image


def sharpen_edges(image, kernel_size=(5, 5)):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened


'''
    Opening-by-reconstruction = Erosion + Morphological reconstruction
'''


def opening_by_reconstruction(image):
    # Erosion
    se = cv2.getStructuringElement(cv2.MORPH_ERODE, (20, 20))  # structure element
    Ie = cv2.erode(image, se, iterations=1)

    # Morphological reconstruction = iteratively dilation
    Iobr = reconstruction(Ie, image, method='dilation')
    Iobr = Iobr.astype('uint8')
    return Iobr


def histogram_equalization(image, mode="clahe"):
    if mode == "normal":
        result = cv2.equalizeHist(image)
        return result
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(image)
        return result

    return None


def create_mask_contour(contours, shape):
    mask_contour = np.zeros(shape)
    mask_contour = mask_contour.astype('int')
    for idx, contour in enumerate(contours):
        for point in contour:
            x = point[0][1]
            y = point[0][0]
            mask_contour[x][y] = idx + 1
    return mask_contour


'''
    Assumption: object in the center of the image
'''


def create_overlapping_image(image_1, image_2, contour_2, shift_x=10, shift_y=10, rotation_angle=90):
    _, mask_2 = contours_to_connected_components(image_2, contour_2)
    mask_2 = mask_2.astype('float32')
    width_2, height_2 = image_2.shape

    # Translation
    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shift_image_2 = cv2.warpAffine(image_2, M=shift_matrix, dsize=image_2.shape, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    shift_mask_2 = cv2.warpAffine(mask_2, M=shift_matrix, dsize=mask_2.shape, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    # Rotation
    rotation_matrix = cv2.getRotationMatrix2D(center=(height_2 / 2, width_2 / 2), angle=rotation_angle, scale=1)
    rotated_image_2 = cv2.warpAffine(shift_image_2, M=rotation_matrix, dsize=image_2.shape,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
    rotated_mask_2 = cv2.warpAffine(shift_mask_2, M=rotation_matrix, dsize=mask_2.shape, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

    overlapping_image = image_1.copy()
    overlapping_image[rotated_mask_2 == 1] = rotated_image_2[rotated_mask_2 == 1]

    return overlapping_image


def get_equalized_image_with_contours(image):
    equalized = histogram_equalization(image)

    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    object_contours = list()
    contour_areas = list()
    for contour in contours:
        area = cv2.contourArea(contour)
        contour_areas.append(area)

    max_area = max(contour_areas)
    for i in range(len(contour_areas)):
        if 50 < contour_areas[i] < max_area:
            object_contours.append(contours[i])

    contour_image = get_contour_image(object_contours, image.shape)
    object_without_contour_image = equalized.copy()
    object_without_contour_image[contour_image == 0] = 0

    equalized_with_contours = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    equalized_with_contours[:, :, 0] = object_without_contour_image
    equalized_with_contours[:, :, 1] = object_without_contour_image

    return equalized_with_contours


def get_center_sub_image(gray, size=256):
    height, width = gray.shape
    cut_height = int((height - size) / 2)
    cut_width = int((width - size) / 2)
    result = gray[cut_height: cut_height + size, cut_width: cut_width + size]
    return result


def get_image_with_points(image, points, radius=9, color=(255, 0, 0), thickness=-1, is_scaled=True):
    height, width = image.shape[:2]
    if len(image.shape) < 3:
        rgb_image = np.stack((image.copy(),) * 3, axis=-1)
    else:
        rgb_image = image.copy()

    for point in points:
        if is_scaled:
            x = int(point[0] * height)
            y = int(point[1] * width)
        else:
            x = point[0]
            y = point[1]
        cv2.circle(rgb_image, center=(x, y), radius=radius, color=color, thickness=thickness)
    return rgb_image


def get_image_with_multiple_points(image, point_set, color_set, radius=9, thickness=-1):
    height, width = image.shape[:2]
    for idx in range(len(point_set)):
        points = point_set[idx]
        for point in points:
            x = int(point[0] * height)
            y = int(point[1] * width)
            cv2.circle(image, center=(x, y), radius=radius, color=color_set[idx], thickness=thickness)
    return image


def rotate_image(image, rotation_angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def translate_image(image, horizontal_shift, vertical_shift):
    rows, cols = image.shape
    M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    shifted_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return shifted_image


def rotate_points(points, rotation_angle, shape=(512, 512)):
    rows, cols = shape[:2]
    M_inv = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    rotated_points = M_inv.dot(points_ones.T).T
    rotated_points = rotated_points.astype('uint16')
    return rotated_points


def translate_points(points, horizontal_shift, vertical_shift):
    M_inv = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    translated_points = M_inv.dot(points_ones.T).T
    return translated_points


def downscale_points(points, shape):
    height, width = shape
    scaled_points = [[x * 1.0 / height, y * 1.0 / width] for x, y in points]
    return np.asarray(scaled_points)


def upscale_points(points, shape):
    height, width = shape
    scaled_points = [[int(x * height), int(y * width)] for x, y in points]
    return np.asarray(scaled_points)


def read_image(image_file, cmap=None):
    if cmap is None or cmap == "rgb":
        image = cv2.imread(image_file)

        if image is None:
            raise Exception("Image is not found.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        image = cv2.imread(image_file, cmap)
        if image is None:
            raise Exception("Image is not found.")
        return image


def find_best_line_brute_fore(binary):
    points = np.argwhere(binary > 0.5)

    dictionary = dict()
    for x, y in points:
        theta = x * 1. / y
        theta = "%.2f" % theta
        if theta not in dictionary:
            dictionary[theta] = [[x, y]]
        else:
            dictionary[theta].append([x, y])

    # for key, value in dictionary.items():
    #     print(key)
    #     print(value)
    #     print(len(value))

    for key, value in sorted(dictionary.items(), key=lambda kv: -len(kv[1])):
        # print(key)
        # print(value)
        return value


def find_best_line_hough_transform(binary):
    # apply probabilistic Hough transform
    minLineLength = 1
    maxLineGap = 1
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 10, minLineLength, maxLineGap)

    return lines[0][0]


def get_block_image(chromosomes, idx, shape=(200, 256, 3)):
    h, w, d = shape
    c_size = int(w / 2)
    c_shape = (c_size, c_size, 3)

    if len(chromosomes) == 0:
        chromosome1 = 255 - np.zeros(c_shape)
        chromosome2 = 255 - np.zeros(c_shape)
    elif len(chromosomes) == 1:
        chromosome1 = chromosomes[0]
        chromosome1 = cv2.resize(chromosome1, c_shape[:2])
        chromosome2 = 255 - np.zeros(c_shape)
    else:
        chromosome1 = chromosomes[0]
        chromosome1 = cv2.resize(chromosome1, c_shape[:2])
        chromosome2 = chromosomes[1]
        chromosome2 = cv2.resize(chromosome2, c_shape[:2])

    image = np.zeros(shape, dtype='uint8')

    image[0: c_size, 0: c_size] = chromosome1.astype('uint8')
    image[0: c_size, c_size: 2 * c_size] = chromosome2.astype('uint8')

    bar_image = read_image(data_dir + "/test/bar.png")
    bar_shape = image[c_size:h, 0:w].shape
    bar_image = cv2.resize(bar_image, (bar_shape[1], bar_shape[0]))

    image[c_size:h, 0:w] = bar_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, idx, (c_size - 10, h - 10), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def get_chromosome_bounding_box(image):
    """

    :param image: chromosome image with dark object and *PURE* white background
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return [x, y, x + w, y + h]


def get_chromosome_contour(image):
    """

    :param image: chromosome image with dark object and *PURE* white background
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]
