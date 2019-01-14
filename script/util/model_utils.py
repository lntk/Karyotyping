import cv2
import numpy as np
import os
from util import general_utils
from os.path import dirname, abspath


def directory_to_data(directory, verbose=True):
    if verbose:
        print("Processing " + directory)
    relative_class_dirs = [x for x in next(os.walk(directory))[1]]
    class_dirs = [directory + "/" + x for x in next(os.walk(directory))[1]]
    class_to_id = {key: value for (value, key) in enumerate(relative_class_dirs)}
    if verbose:
        print(class_to_id)

    id_to_class = {key: value for (key, value) in enumerate(relative_class_dirs)}
    X, y, image_names = list(), list(), list()
    for class_dir in relative_class_dirs:
        class_id = int(class_to_id[class_dir])
        class_dir = directory + "/" + class_dir
        if verbose:
            print("Processing " + class_dir)
        image_files = general_utils.get_all_files(class_dir)
        for image_file in image_files:
            image = cv2.imread(class_dir + "/" + image_file, 0)
            _, _, k_number = image_file[:-4].split("_")
            image_names.append(id_to_class[class_id] + "_karyotyping_" + k_number)
            X.append(image)
            y.append(class_id)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, id_to_class, image_names


def get_num_data(directory):
    count = 0
    class_dirs = [directory + "/" + x for x in next(os.walk(directory))[1]]
    for class_dir in class_dirs:
        image_files = general_utils.get_all_files(class_dir)
        count += len(image_files)

    return count
