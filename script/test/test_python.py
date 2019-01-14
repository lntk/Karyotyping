import unittest
import os
import psutil
import numpy as np
import cv2
from script.util import image_utils


class Test(unittest.TestCase):
    def test_memory_usage(self):
        a = 10
        b = a
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)  # 27627520

    def test_dictionary(self):
        l1 = [1, 2]
        l2 = [3, 4]
        dictionary = {"list 1": l1, "list 2": l2}
        dictionary["list 1"].append(0)
        print(l1)

    def test_array_index_getter(self):
        l = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        a = np.asarray(l)
        print(a)
        print(a[0])
        print(a[:, 1])
        print(a[:, [0, 2, 4]])

    def test_remove_from_list_duplicate(self):
        a = [1, 1, 2, 2, 3, 4]
        a.remove(1)
        print(a)

    def test_range(self):
        print(list(range(1, 3)))

    def test_rewrite_image(self):
        idx = 12
        image = image_utils.read_image("/home/lntk/Desktop/Karyotype/data/pipeline/karyotype_" + str(idx) + ".bmp")
        cv2.imwrite("/home/lntk/Desktop/Karyotype/data/pipeline/karyotype_" + str(idx) + "_clone.bmp", image)
