"""
    @author: lntk
"""

import unittest
from os.path import dirname, abspath
import numpy as np
import cv2

from script.util import image_utils, general_utils, chromosome_utils

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test_generate_chromosome_cluster(self):
        image_dir = data_dir + "/chromosome/1"
        image_files = general_utils.get_all_files(image_dir)
        num_chromosome = 15
        contours = list()

        chromosomes = list()
        chosen_files = list()

        # error_case = ['/home/lntk/Desktop/Karyotype/data/chromosome/1/xx_karyotype_225_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_213_1.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_190_1.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xx_karyotype_064_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xx_karyotype_017_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_055_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_208_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_266_1.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_132_0.bmp',
        #               '/home/lntk/Desktop/Karyotype/data/chromosome/1/xy_karyotype_233_1.bmp']

        for _ in range(num_chromosome):
            """ Randomly get chromosome image from directory """
            idx = np.random.randint(len(image_files))
            chromosome = image_utils.read_image(image_dir + "/" + image_files[idx])
            chosen_files.append(image_dir + "/" + image_files[idx])
        # for image_file in error_case:
        #     chromosome = image_utils.read_image(image_file)

            """ Resize """
            chromosome = cv2.resize(chromosome, (64, 64))

            """ Rotate """
            angle = np.random.randint(90)
            rotated_chromosome = image_utils.rotate_image(chromosome, angle)

            """ Extract contour """
            contour = image_utils.get_chromosome_contour(rotated_chromosome)
            contours.append(contour)

            chromosomes.append(rotated_chromosome)

        print(chosen_files)

        contours, boxes, initial_boxes = chromosome_utils.generate_chromosome_cluster(contours)

        for contour in contours:
            print(contour.shape)

        # silhouette_image = chromosome_utils.get_chromosome_silhouette(contours, boxes)
        cluster_image = chromosome_utils.get_chromosome_cluster_image(boxes, initial_boxes, chromosomes)
        image_utils.show_multiple_images([cluster_image])
