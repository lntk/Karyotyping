from script.util import general_utils
import numpy as np
from os.path import dirname, abspath
import cv2

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


def get_test_image():
    chromosome_dir = data_dir + "/chromosome"
    chromosome_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "x", "y"]
    idx = np.random.randint(len(chromosome_ids))
    chromosome_dir = chromosome_dir + "/" + str(chromosome_ids[idx])
    image_files = general_utils.get_all_files(chromosome_dir)
    image_file = image_files[np.random.randint(len(image_files))]
    test_image = cv2.imread(chromosome_dir + "/" + image_file, 0)
    return test_image
