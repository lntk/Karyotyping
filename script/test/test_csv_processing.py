import unittest
from os.path import dirname, abspath

from script.processing import csv_processing
from script.util import data_utils

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test_image_files_to_bounding_box_csv(self):
        directory = data_dir + "/chromosome"
        csv_dir = data_dir + "/test/csv"

        image_files, labels, _, _ = data_utils.directory_to_images_files_and_labels(directory)
        image_files = image_files[:10]
        labels = labels[:10]
        csv_processing.image_files_to_bounding_box_csv(image_files, labels, csv_dir)

    def test(self):
        csv_dir = data_dir + "/test/csv"
        class_mapping = [[str(i), i] for i in range(24)]
        class_mapping_file = csv_dir + "/" + "" + "class_mapping.csv"
        from script.util import general_utils
        general_utils.write_list_to_csv(class_mapping, class_mapping_file)
