import unittest
from script.util import general_utils, image_utils, data_utils
import numpy as np
from os.path import dirname, abspath
import cv2

working_dir = dirname(dirname(dirname(__file__)))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    @staticmethod
    def get_test_image():
        chromosome_dir = data_dir + "/chromosome"
        chromosome_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "x", "y"]
        idx = np.random.randint(len(chromosome_ids))
        chromosome_dir = chromosome_dir + "/" + str(chromosome_ids[idx])
        image_files = general_utils.get_all_files(chromosome_dir)
        image_file = image_files[np.random.randint(len(image_files))]
        test_image = cv2.imread(chromosome_dir + "/" + image_file, 0)
        return test_image

    def test_rotate_points(self):
        rotation_angle = 30
        points = np.asarray([[100., 200.]])
        test_image = self.get_test_image()

        shape = test_image.shape

        rotated_points = image_utils.rotate_points(points, rotation_angle=rotation_angle)
        rotated_test_image = image_utils.rotate_image(test_image, rotation_angle=rotation_angle)

        scaled_points = image_utils.downscale_points(points, shape=shape)
        scaled_rotated_points = image_utils.downscale_points(rotated_points, shape=shape)

        test_image_points = image_utils.get_image_with_points(test_image, scaled_points)
        rotated_test_image_points = image_utils.get_image_with_points(rotated_test_image, scaled_rotated_points)
        image_utils.show_multiple_images([test_image_points, rotated_test_image_points])

    def test_translate_points(self):
        horizontal_shift = 50
        vertical_shift = 40
        points = np.asarray([[100., 200.]])
        test_image = self.get_test_image()

        shape = test_image.shape

        translated_points = image_utils.translate_points(points, horizontal_shift=horizontal_shift,
                                                         vertical_shift=vertical_shift)
        translated_test_image = image_utils.translate_image(test_image, horizontal_shift=horizontal_shift,
                                                            vertical_shift=vertical_shift)

        print(points)
        print(translated_points)

        scaled_points = image_utils.downscale_points(points, shape=shape)
        scaled_translated_points = image_utils.downscale_points(translated_points, shape=shape)

        test_image_points = image_utils.get_image_with_points(test_image, scaled_points)
        translated_test_image_points = image_utils.get_image_with_points(translated_test_image,
                                                                         scaled_translated_points)
        image_utils.show_multiple_images([test_image_points, translated_test_image_points])

    def test_rotate_and_translate(self):
        rotation_angle = 30
        horizontal_shift = 50
        vertical_shift = 40
        points = np.asarray([[100., 200.]])
        test_image = self.get_test_image()

        shape = test_image.shape

        rotated_points = image_utils.rotate_points(points, rotation_angle=-rotation_angle)
        rotated_test_image = image_utils.rotate_image(test_image, rotation_angle=rotation_angle)
        translated_points = image_utils.translate_points(rotated_points, horizontal_shift=horizontal_shift,
                                                         vertical_shift=vertical_shift)
        translated_test_image = image_utils.translate_image(rotated_test_image, horizontal_shift=horizontal_shift,
                                                            vertical_shift=vertical_shift)

        scaled_points = image_utils.downscale_points(points, shape=shape)
        scaled_translated_points = image_utils.downscale_points(translated_points, shape=shape)

        test_image_points = image_utils.get_image_with_points(test_image, scaled_points)
        translated_test_image_points = image_utils.get_image_with_points(translated_test_image,
                                                                         scaled_translated_points)
        image_utils.show_multiple_images([test_image_points, translated_test_image_points])

    def test_remove_chromosomes(self):
        image = image_utils.read_image(data_dir + "/test/karyotyping_1.bmp")
        image = data_utils.remove_chromosomes(image)
        cv2.imwrite(data_dir + "/test/karyotyping_frame.bmp", image)

    def test_get_block_image(self):
        chromosome1 = image_utils.read_image(data_dir + "/chromosome/1/xx_karyotype_001_0.bmp")
        chromosome2 = image_utils.read_image(data_dir + "/chromosome/1/xx_karyotype_001_1.bmp")
        image = image_utils.get_block_image([chromosome1, chromosome2], "1")
        image_utils.show_image(image)

